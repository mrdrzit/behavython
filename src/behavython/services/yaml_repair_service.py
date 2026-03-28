from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class YamlRepairResult:
    success: bool
    original_path: Path
    repaired_path: Path | None
    changed: bool
    message: str
    config: dict[str, Any] | None = None


def _looks_like_broken_video_key_start(line: str) -> bool:
    stripped = line.strip()

    if not stripped:
        return False

    if stripped.startswith("crop:"):
        return False

    if stripped.endswith(":"):
        return False

    has_windows_path = ":\\" in stripped or "\\" in stripped
    return has_windows_path


def _looks_like_video_key_continuation(line: str) -> bool:
    stripped = line.strip().lower()
    return (
        stripped.endswith(".mp4:")
        or stripped.endswith(".avi:")
        or stripped.endswith(".mov:")
        or stripped.endswith(".mkv:")
    )


def _repair_multiline_video_keys(raw_text: str) -> tuple[str, bool]:
    lines = raw_text.splitlines()
    repaired_lines: list[str] = []
    changed = False
    inside_video_sets = False
    pending_prefix: str | None = None
    pending_indent: str = ""

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("video_sets:"):
            inside_video_sets = True
            repaired_lines.append(line)
            continue

        if inside_video_sets and stripped and not line.startswith((" ", "\t")):
            inside_video_sets = False

        if inside_video_sets:
            if pending_prefix is not None:
                if _looks_like_video_key_continuation(line):
                    merged = f"{pending_indent}{pending_prefix}{stripped}"
                    repaired_lines.append(merged)
                    pending_prefix = None
                    pending_indent = ""
                    changed = True
                    continue
                else:
                    repaired_lines.append(f"{pending_indent}{pending_prefix.rstrip()}")
                    pending_prefix = None
                    pending_indent = ""

            if _looks_like_broken_video_key_start(line):
                pending_indent = line[: len(line) - len(line.lstrip())]
                pending_prefix = stripped + " "
                continue

        repaired_lines.append(line)

    if pending_prefix is not None:
        repaired_lines.append(f"{pending_indent}{pending_prefix.rstrip()}")

    repaired_text = "\n".join(repaired_lines)
    if raw_text.endswith("\n"):
        repaired_text += "\n"

    return repaired_text, changed


def validate_yaml_text(yaml_text: str) -> dict[str, Any]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError("YAML root is not a dictionary.")
    return data


def validate_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        return validate_yaml_text(handle.read())


def repair_dlc_config_yaml(
    source_path: str | Path,
    destination_path: str | Path | None = None,
    overwrite: bool = False,
) -> YamlRepairResult:
    source = Path(source_path)

    if not source.exists():
        return YamlRepairResult(
            success=False,
            original_path=source,
            repaired_path=None,
            changed=False,
            message=f"File does not exist: {source}",
            config=None,
        )

    raw_text = source.read_text(encoding="utf-8")
    repaired_text, changed = _repair_multiline_video_keys(raw_text)

    try:
        parsed = validate_yaml_text(repaired_text)
    except Exception as exc:
        return YamlRepairResult(
            success=False,
            original_path=source,
            repaired_path=None,
            changed=changed,
            message=f"Repair attempt failed validation: {exc}",
            config=None,
        )

    if overwrite:
        out_path = source
    elif destination_path is not None:
        out_path = Path(destination_path)
    else:
        out_path = source.with_name(f"{source.stem}_repaired{source.suffix}")

    out_path.write_text(repaired_text, encoding="utf-8")

    return YamlRepairResult(
        success=True,
        original_path=source,
        repaired_path=out_path,
        changed=changed,
        message="YAML repaired and validated successfully." if changed else "YAML validated successfully.",
        config=parsed,
    )


def load_or_repair_dlc_yaml(path: str | Path) -> tuple[dict[str, Any], Path, bool]:
    source = Path(path)

    try:
        data = validate_yaml_file(source)
        return data, source, False
    except Exception:
        result = repair_dlc_config_yaml(source)
        if not result.success or result.config is None or result.repaired_path is None:
            raise
        return result.config, result.repaired_path, result.changed