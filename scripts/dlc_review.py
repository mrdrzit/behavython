"""
Reviews DeepLabCut labeled-data folders of any project, one napari window per folder.

Each folder opens in a fresh napari process with the DeepLabCut keypoint controls, the keypoints
layer selected and the "select points" tool active. Correct, press Ctrl+S, close the window: the
next folder opens. A folder counts as reviewed when its CollectedData file was saved while its window
was open; if you close without saving, the terminal asks what to do.

Folders to review (combine freely; default = every folder in the target's labeled-data):
  names          folders already inside <target>/labeled-data, e.g. TR_A1_2026-06-21_09-30-00
  paths          folders anywhere else (another project, a pre-labeling output, ...). They are imported
                 first: copied into <target>/labeled-data, with labels converted to the target's
                 scorer and bodyparts. Existing folders in the target are never overwritten.
  --list FILE    text file with one name or path per line
  --match GLOB   only folders whose name matches, e.g. "TT_*2026-07*"

Progress is kept in <target>/review_log.csv, so re-running resumes where you stopped.
napari-deeplabcut can only save folders that sit inside a project's labeled-data, hence the import.
"""

import argparse
import csv
import fnmatch
import shutil
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def read_project(config_path: Path) -> dict:
    import yaml

    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get("multianimalproject"):
        raise SystemExit("Multi-animal projects are not supported.")
    return {"root": config_path.parent, "scorer": cfg["scorer"], "bodyparts": list(cfg["bodyparts"])}


def import_folder(source: Path, project: dict) -> Path | None:
    """Copies a labeled folder into the target's labeled-data, converting its labels to the target format."""
    import pandas as pd

    destination = project["root"] / "labeled-data" / source.name
    if destination.exists():
        print(f"  already in target, using the existing copy: {source.name}")
        return destination
    images = [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        print(f"  skip (no images): {source}")
        return None

    labels = None
    h5_files = sorted(source.glob("CollectedData_*.h5"))
    if h5_files:
        df = pd.read_hdf(h5_files[0])
        missing = [bp for bp in project["bodyparts"] if bp not in df.columns.get_level_values("bodyparts")]
        if missing:
            print(f"  {source.name}: bodyparts not in source, left empty: {missing}")
        columns = pd.MultiIndex.from_product([[project["scorer"]], project["bodyparts"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])
        values = {}
        for _, bodypart, coord in columns:
            match = [c for c in df.columns if c[1] == bodypart and c[2] == coord]
            values[(project["scorer"], bodypart, coord)] = df[match[0]] if match else float("nan")
        labels = pd.DataFrame({column: values[column] for column in columns}, index=df.index)
        labels.columns = columns
        image_names = [Path(str(idx[-1] if isinstance(idx, tuple) else idx)).name for idx in df.index]
        labels.index = pd.MultiIndex.from_arrays([["labeled-data"] * len(labels), [source.name] * len(labels), image_names])
        labels = labels[labels.index.get_level_values(2).isin({p.name for p in images})]

    destination.mkdir(parents=True)
    for image in images:
        shutil.copy2(image, destination / image.name)
    if labels is not None:
        labels.to_hdf(destination / f"CollectedData_{project['scorer']}.h5", key="df_with_missing", mode="w")
        labels.to_csv(destination / f"CollectedData_{project['scorer']}.csv")
    print(f"  imported: {source} -> labeled-data/{source.name}" + ("" if labels is not None else " (no labels, starts empty)"))
    return destination


def resolve_folders(project: dict, entries: list[str], pattern: str | None) -> list[Path]:
    labeled_data = project["root"] / "labeled-data"
    if not entries:
        entries = sorted(p.name for p in labeled_data.iterdir() if p.is_dir())
    folders = []
    for entry in entries:
        path = Path(entry)
        if not path.is_absolute() or path.parent == labeled_data:
            folder = labeled_data / path.name
            if not folder.is_dir():
                print(f"  skip (not in labeled-data): {entry}")
                continue
        else:
            folder = import_folder(path, project) if path.is_dir() else None
            if folder is None:
                continue
        if pattern and not fnmatch.fnmatch(folder.name, pattern):
            continue
        if not any(p.suffix.lower() in IMAGE_SUFFIXES for p in folder.iterdir()):
            print(f"  skip (no images): {folder.name}")
            continue
        if folder not in folders:
            folders.append(folder)
    return folders


def read_log(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    with open(log_path, encoding="utf-8") as handle:
        return {row["folder"] for row in csv.DictReader(handle)}


def append_log(log_path: Path, tag: str, folder: Path):
    is_new = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if is_new:
            writer.writerow(["round", "folder", "reviewed_at"])
        writer.writerow([tag, folder.name, datetime.now().isoformat(timespec="seconds")])


def open_single(folder: Path, config: Path, title: str):
    """Child process: one viewer, one folder, exits when the window is closed."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    import napari
    from napari.layers import Points
    from qtpy.QtCore import QTimer

    viewer = napari.Viewer(title=title)
    viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")
    viewer.open(str(folder), plugin="napari-deeplabcut")
    if not any(isinstance(layer, Points) for layer in viewer.layers):
        # Unlabeled folder: the config provides an empty keypoints layer to annotate
        viewer.open(str(config), plugin="napari-deeplabcut")

    def start_in_select_mode():
        points = [layer for layer in viewer.layers if isinstance(layer, Points)]
        if points:
            viewer.layers.selection.active = points[0]
            points[0].mode = "select"

    # Once now and once after the plugin's deferred setup, which may switch the layer to "add"
    start_in_select_mode()
    QTimer.singleShot(500, start_in_select_mode)
    napari.run()


def label_mtime(folder: Path, scorer: str) -> float:
    label_file = folder / f"CollectedData_{scorer}.h5"
    return label_file.stat().st_mtime if label_file.exists() else 0.0


def review(config: Path, folders: list[Path], scorer: str, tag: str, include_reviewed: bool):
    log_path = config.parent / "review_log.csv"
    reviewed = read_log(log_path)
    queue = [f for f in folders if include_reviewed or f.name not in reviewed]
    print(f"{len(folders)} folders, {len(folders) - len(queue)} already reviewed, {len(queue)} to go.")

    for position, folder in enumerate(queue, start=1):
        done = len(read_log(log_path) & {f.name for f in folders})
        title = f"Review [{position}/{len(queue)}] (reviewed {done}/{len(folders)}) - {folder.name}"
        while True:
            print(f"\n[{position}/{len(queue)}] {folder.name}: correct, Ctrl+S, close the window.")
            before = label_mtime(folder, scorer)
            subprocess.run([sys.executable, __file__, "--single", str(folder), str(config), title], check=False)
            if label_mtime(folder, scorer) > before:
                append_log(log_path, tag, folder)
                break
            answer = input("Not saved. [r]eopen, mark as [o]k without changes, [s]kip, [q]uit: ").strip().lower()
            if answer == "o":
                append_log(log_path, tag, folder)
                break
            if answer == "s":
                break
            if answer == "q":
                return
    print(f"\n{len(read_log(log_path) & {f.name for f in folders})}/{len(folders)} folders reviewed.")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--single":
        open_single(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
        return

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("config", help="Target project's config.yaml.")
    parser.add_argument("folders", nargs="*", help="Folder names in labeled-data, or paths to import (default: all of labeled-data).")
    parser.add_argument("--list", help="Text file with one folder name or path per line.")
    parser.add_argument("--match", help='Only folders whose name matches this glob, e.g. "TT_*".')
    parser.add_argument("--tag", default="review", help="Label written to review_log.csv for this session.")
    parser.add_argument("--all", action="store_true", help="Also open folders already marked as reviewed.")
    parser.add_argument("--dry-run", action="store_true", help="Import and list the folders, do not open napari.")
    args = parser.parse_args()

    config = Path(args.config)
    project = read_project(config)
    entries = list(args.folders)
    if args.list:
        entries += [line.strip() for line in open(args.list, encoding="utf-8") if line.strip() and not line.startswith("#")]
    folders = resolve_folders(project, entries, args.match)
    if not folders:
        raise SystemExit("No folders to review.")
    if args.dry_run:
        reviewed = read_log(config.parent / "review_log.csv")
        for folder in folders:
            print(f"  {'[reviewed] ' if folder.name in reviewed else ''}{folder.name}")
        return
    review(config, folders, project["scorer"], args.tag, args.all)


if __name__ == "__main__":
    main()
