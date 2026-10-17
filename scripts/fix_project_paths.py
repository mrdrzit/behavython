"""
Points a DeepLabCut project's stored absolute paths at the folder it is in now. Run it after moving or
copying a project to another drive or machine. Uses only the standard library.

Fields rewritten (everything else, including the video_sets keys, is left alone):
  config.yaml                                   project_path
  dlc-models-pytorch/**/train/pytorch_config*   metadata.project_path, metadata.pose_config_path
  dlc-models-pytorch/**/test/pose_cfg.yaml      dataset
video_sets keys are the original video locations. Training does not read them, but DLC analysis may
need them to exist, so they are not changed.

  python fix_project_paths.py [project folder] [--dry-run]

With no folder it uses the folder this script sits in, or the current folder if that has a config.yaml.
"""

import argparse
import re
from pathlib import Path


def find_project(argument: str | None) -> Path:
    candidates = [Path(argument)] if argument else [Path(__file__).resolve().parent, Path.cwd()]
    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate.resolve()
    raise SystemExit("No config.yaml found. Give the project folder as an argument.")


def rewrite(path: Path, rules: dict[str, str], dry_run: bool) -> int:
    """rules maps a yaml key to its new value; matches `key: value` lines at any indentation."""
    text = path.read_text(encoding="utf-8")
    changed = 0
    lines = text.splitlines(keepends=True)
    for number, line in enumerate(lines):
        match = re.match(r"^(\s*)([A-Za-z_]+):\s*(.*?)(\r?\n?)$", line)
        if not match or match.group(2) not in rules:
            continue
        indent, key, old, ending = match.groups()
        new = rules[key](old) if callable(rules[key]) else rules[key]
        if new != old.strip("'\""):
            print(f"  {path.name}:{number + 1} {key}: {old}  ->  {new}")
            lines[number] = f"{indent}{key}: {new}{ending}"
            changed += 1
    if changed and not dry_run:
        path.write_text("".join(lines), encoding="utf-8")
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("project", nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = find_project(args.project)
    print(f"project: {root}")

    def relocate(old: str) -> str:
        """Keeps the part of an old path from dlc-models-pytorch onwards, under the new root."""
        parts = re.split(r"[\\/]", old.strip("'\""))
        if "dlc-models-pytorch" not in parts:
            return old
        return str(root.joinpath(*parts[parts.index("dlc-models-pytorch") :]))

    total = rewrite(root / "config.yaml", {"project_path": str(root)}, args.dry_run)
    models = root / "dlc-models-pytorch"
    for file in sorted(models.rglob("pytorch_config*")) if models.exists() else []:
        total += rewrite(file, {"project_path": str(root), "pose_config_path": relocate}, args.dry_run)
    for file in sorted(models.rglob("pose_cfg.yaml")) if models.exists() else []:
        total += rewrite(file, {"dataset": str(root)}, args.dry_run)
    print(f"{total} field(s) {'would change' if args.dry_run else 'changed'}.")


if __name__ == "__main__":
    main()
