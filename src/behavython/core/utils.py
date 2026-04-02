from __future__ import annotations

import os
import re
from typing import Any
from pathlib import Path
from natsort import os_sorted
from collections import defaultdict
from src.behavython.core.defaults import VALID_VIDEO_EXTENSIONS
from src.behavython.services.validation import validate_yaml_file, validate_yaml_text
from src.behavython.pipeline.models import (
    AnalysisInputSource,
    OutputFolderSource,
    ResolvedAnalysisInput,
    ResolvedOutputFolder,
    ResolvedVideoInput,
    VideoInputSource,
    YamlRepairResult,
)


def _normalize_path_text(value: str) -> str:
    cleaned_value = value.strip()

    if not cleaned_value:
        return ""

    cleaned_value = cleaned_value.rstrip(",").strip('"').strip("'").strip()
    return cleaned_value


def _deduplicate_preserving_order(paths: list[str]) -> tuple[list[str], list[str]]:
    unique_paths: list[str] = []
    duplicate_entries: list[str] = []
    seen_paths: set[str] = set()

    for path in paths:
        if path in seen_paths:
            duplicate_entries.append(path)
            continue

        seen_paths.add(path)
        unique_paths.append(path)

    return unique_paths, duplicate_entries


def _collect_videos_from_folder(folder_path: str) -> list[str]:
    if not folder_path or not os.path.isdir(folder_path):
        return []

    video_paths: list[str] = []

    for entry_name in os.listdir(folder_path):
        entry_path = os.path.join(folder_path, entry_name)

        if os.path.isfile(entry_path) and entry_name.lower().endswith(VALID_VIDEO_EXTENSIONS):
            video_paths.append(entry_path)

    return sorted(video_paths)


def _collect_videos_from_txt(txt_path: str) -> tuple[list[str], list[str]]:
    if not txt_path or not os.path.isfile(txt_path):
        return [], []

    resolved_paths: list[str] = []
    skipped_entries: list[str] = []

    with open(txt_path, "r", encoding="utf-8") as handle:
        for raw_line in handle.readlines():
            raw_entry = raw_line.rstrip("\n\r")
            cleaned_entry = _normalize_path_text(raw_entry)

            if not cleaned_entry:
                if raw_entry.strip():
                    skipped_entries.append(raw_entry)
                continue

            resolved_paths.append(cleaned_entry)

    return resolved_paths, skipped_entries


def resolve_video_input(source: VideoInputSource) -> ResolvedVideoInput:
    folder_path = _normalize_path_text(source.folder_path)
    txt_path = _normalize_path_text(source.txt_path)

    warnings: list[str] = []
    resolved_paths: list[str] = []
    skipped_entries: list[str] = []
    source_kind: str | None = None

    if txt_path:
        if not os.path.exists(txt_path):
            warnings.append(f"Selected txt file does not exist: {txt_path}")
        elif not txt_path.lower().endswith(".txt"):
            warnings.append(f"Selected file is not a .txt file: {txt_path}")
        else:
            resolved_paths, skipped_entries = _collect_videos_from_txt(txt_path)
            source_kind = "txt"

            if not resolved_paths:
                warnings.append("No video paths were found in the selected txt file.")

    elif folder_path:
        if not os.path.isdir(folder_path):
            warnings.append(f"Selected video folder does not exist: {folder_path}")
        else:
            resolved_paths = _collect_videos_from_folder(folder_path)
            source_kind = "folder"

            if not resolved_paths:
                warnings.append("No valid videos were found in the selected folder.")

    unique_paths, duplicate_entries = _deduplicate_preserving_order(resolved_paths)

    return ResolvedVideoInput(
        paths=unique_paths,
        source_kind=source_kind,
        skipped_entries=skipped_entries,
        duplicate_entries=duplicate_entries,
        warnings=warnings,
    )


def resolve_analysis_input(source: AnalysisInputSource) -> ResolvedAnalysisInput:
    cleaned_paths: list[str] = []
    skipped_entries: list[str] = []

    for path in source.selected_files:
        cleaned_path = _normalize_path_text(path)

        if not cleaned_path:
            skipped_entries.append(path)
            continue

        cleaned_paths.append(cleaned_path)

    unique_paths, duplicate_entries = _deduplicate_preserving_order(cleaned_paths)

    warnings: list[str] = []
    if duplicate_entries:
        warnings.append("Duplicate analysis files were removed.")

    return ResolvedAnalysisInput(
        paths=unique_paths,
        skipped_entries=skipped_entries,
        duplicate_entries=duplicate_entries,
        warnings=warnings,
    )


def resolve_output_folder(source: OutputFolderSource) -> ResolvedOutputFolder:
    folder_path = _normalize_path_text(source.selected_folder)

    warnings: list[str] = []
    if source.selected_folder and not folder_path:
        warnings.append("Selected folder path became empty after normalization.")

    return ResolvedOutputFolder(
        path=folder_path,
        warnings=warnings,
    )


def group_analysis_files(file_paths: list[str]) -> list[dict]:
    file_paths = os_sorted(file_paths)

    def classify_file(name_lower: str) -> str:
        if "dlc" in name_lower and "_filtered" in name_lower:
            return "skeleton" if "_filtered_skeleton" in name_lower else "position"

        if re.search(r"roi[lr]?\.csv$", name_lower):
            return "roi"

        if name_lower.endswith((".jpg", ".jpeg", ".png")):
            return "image"

        if name_lower.endswith(".mp4"):
            return "video"

        return "unknown"

    def extract_raw_id(name: str, name_lower: str, file_type: str) -> str | None:
        """
        Extracts the base ID from the filename while strictly preserving the original casing.
        """
        if file_type == "unknown":
            return None

        if file_type in ("position", "skeleton"):
            idx = name_lower.find("dlc")
            if idx != -1:
                return name[:idx]

        elif file_type == "roi":
            return re.sub(r"_?roi[lr]?\.csv$", "", name, flags=re.IGNORECASE)

        elif file_type in ("image", "video"):
            return name.rsplit(".", 1)[0]

        return None

    groups = defaultdict(
        lambda: {
            "animal_id": None,
            "files": {
                "position": [],
                "skeleton": [],
                "roi": [],
                "image": [],
                "video": [],
            },
        }
    )

    for path in file_paths:
        name = os.path.basename(path)
        name_lower = name.lower()

        file_type = classify_file(name_lower)
        raw_id = extract_raw_id(name, name_lower, file_type)

        if not raw_id:
            continue

        display_id = raw_id.rstrip("_")

        group_key = display_id.lower()
        group = groups[group_key]

        if group["animal_id"] is None:
            group["animal_id"] = display_id

        group["files"][file_type].append(path)

    return list(groups.values())


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
    return stripped.endswith(".mp4:") or stripped.endswith(".avi:") or stripped.endswith(".mov:") or stripped.endswith(".mkv:")


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
