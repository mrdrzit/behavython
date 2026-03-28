from __future__ import annotations

import os

from behavython.config.defaults import VALID_VIDEO_EXTENSIONS
from behavython.shared.models import (
    AnalysisInputSource,
    OutputFolderSource,
    ResolvedAnalysisInput,
    ResolvedOutputFolder,
    ResolvedVideoInput,
    VideoInputSource,
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