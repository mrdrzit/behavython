from __future__ import annotations

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Any
from behavython.core.defaults import VALID_VIDEO_EXTENSIONS
from behavython.core.paths import FFMPEG_BIN_DIR
from behavython.core.utils import load_or_repair_dlc_yaml
from behavython.services.validation import validate_config_path, validate_video_paths
from behavython.services.logging import capture_external_output
from behavython.pipeline.models import (
    DLCClearUnusedFilesRequest,
    DLCFrameExtractionRequest,
    DLCSkeletonExtractionRequest,
    DLCVideoAnalysisRequest,
)

app_logger = logging.getLogger("behavython")
dlc_logger = logging.getLogger("behavython.dlc")
console_logger = logging.getLogger("behavython.console")


def load_deeplabcut():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MPLBACKEND", "Agg")

    if "deeplabcut" not in sys.modules:
        import deeplabcut  # type: ignore
    else:
        deeplabcut = sys.modules["deeplabcut"]
    return deeplabcut


def prepare_dlc_config(config_path: str) -> tuple[dict[str, Any], str, bool]:
    """
    Returns:
        config_dict,
        usable_config_path,
        was_repaired
    """
    config_dict, resolved_path, was_repaired = load_or_repair_dlc_yaml(config_path)
    return config_dict, str(Path(resolved_path)), was_repaired


def _emit_config_repair_logs(original_path: str, usable_path: str, was_repaired: bool, log) -> None:
    if not log:
        return

    if was_repaired:
        log.emit("dlc", "Broken config detected and repaired.")
        log.emit("dlc", f"Original config: {original_path}")
        log.emit("dlc", f"Using repaired config: {usable_path}")
    else:
        log.emit("dlc", "Config.yaml validated successfully.")


def run_dlc_video_analysis(request: DLCVideoAnalysisRequest, progress=None, log=None, warning=None):
    errors = validate_config_path(request.config_path) + validate_video_paths(request.video_paths)
    if errors:
        raise ValueError("\n".join(errors))

    dlc_logger.info(
        "Deeplabcut video analysis started for %d video(s). config=%s",
        len(request.video_paths),
        request.config_path,
    )

    if progress:
        progress.emit(20)

    _, usable_config_path, was_repaired = prepare_dlc_config(request.config_path)
    _emit_config_repair_logs(request.config_path, usable_config_path, was_repaired, log)

    deeplabcut = load_deeplabcut()

    if log:
        log.emit("dlc", f"Using DeepLabCut version {deeplabcut.__version__}")

    extension = os.path.splitext(request.video_paths[0])[1].lower()

    if log:
        log.emit("dlc", "Analyzing videos...")
    dlc_logger.info("Calling deeplabcut.analyze_videos")

    if progress:
        progress.emit(40)

    console_logger.info("Starting video analysis with DeepLabCut for %d video(s)", len(request.video_paths))
    with capture_external_output("behavython.external"):
        deeplabcut.analyze_videos(
            usable_config_path,
            request.video_paths,
            videotype=extension,
            shuffle=1,
            trainingsetindex=0,
            gputouse=0,
            allow_growth=True,
            save_as_csv=True,
        )

    if log:
        log.emit("dlc", "Filtering predictions...")
    dlc_logger.info("Calling deeplabcut.filterpredictions")

    if progress:
        progress.emit(60)

    console_logger.info(f"\nFiltering predictions with DeepLabCut for {len(request.video_paths)} video(s)")
    with capture_external_output("behavython.external"):
        deeplabcut.filterpredictions(
            usable_config_path,
            request.video_paths,
            videotype=extension,
            shuffle=1,
            trainingsetindex=0,
            filtertype="median",
            save_as_csv=True,
        )

    if request.create_plots:
        if log:
            log.emit("dlc", "Generating trajectory plots...")
        dlc_logger.info("Calling deeplabcut.plot_trajectories")

        if progress:
            progress.emit(80)

        console_logger.info("Generating trajectory plots with DeepLabCut for %d video(s)", len(request.video_paths))
        with capture_external_output("behavython.external"):
            deeplabcut.plot_trajectories(
                usable_config_path,
                request.video_paths,
                videotype=extension,
                showfigures=False,
                filtered=True,
            )

    if progress:
        progress.emit(100)

    dlc_logger.info("run_dlc_video_analysis finished successfully")

    return {
        "kind": "dlc_analysis",
        "videos": len(request.video_paths),
        "config_path_used": usable_config_path,
        "config_was_repaired": was_repaired,
    }


def run_extract_skeleton(request: DLCSkeletonExtractionRequest, progress=None, log=None, warning=None):
    errors = validate_config_path(request.config_path) + validate_video_paths(request.video_paths)
    if errors:
        raise ValueError("\n".join(errors))

    dlc_logger.info(
        "Skeleton extraction started for %d video(s). config=%s",
        len(request.video_paths),
        request.config_path,
    )

    _, usable_config_path, was_repaired = prepare_dlc_config(request.config_path)
    _emit_config_repair_logs(request.config_path, usable_config_path, was_repaired, log)

    deeplabcut = load_deeplabcut()
    extension = os.path.splitext(request.video_paths[0])[1].lower()

    if log:
        log.emit("dlc", f"Using DeepLabCut version {deeplabcut.__version__}")
        log.emit("dlc", "Extracting skeleton...")

    for index, video in enumerate(request.video_paths, start=1):
        console_logger.info("Extracting skeleton for video %d/%d: %s", index, len(request.video_paths), video)
        with capture_external_output("behavython.external"):
            deeplabcut.filterpredictions(
                usable_config_path,
                video,
                videotype=extension,
                shuffle=1,
                trainingsetindex=0,
                filtertype="median",
                save_as_csv=True,
            )
            deeplabcut.analyzeskeleton(
                usable_config_path,
                video,
                shuffle=1,
                trainingsetindex=0,
                filtered=True,
                save_as_csv=True,
            )

        if progress:
            progress.emit(round((index / len(request.video_paths)) * 100))

    dlc_logger.info("run_extract_skeleton finished successfully")

    return {
        "kind": "dlc_skeleton",
        "videos": len(request.video_paths),
        "config_path_used": usable_config_path,
        "config_was_repaired": was_repaired,
    }


def _get_video_fps(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, cwd=FFMPEG_BIN_DIR, capture_output=True, text=True, check=True)
    numerator, denominator = map(float, result.stdout.strip().split("/"))
    return numerator / denominator


def _get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, cwd=FFMPEG_BIN_DIR, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def run_extract_frames(request: DLCFrameExtractionRequest, progress=None, log=None, warning=None):
    errors = validate_video_paths(request.video_paths)
    if errors:
        raise ValueError("\n".join(errors))

    total = len(request.video_paths)

    for index, video_path in enumerate(request.video_paths, start=1):
        fps = _get_video_fps(video_path)
        duration = _get_video_duration(video_path)

        frame_number = request.override_frame_number
        if frame_number is None:
            frame_number = int((duration * fps) * 0.5)

        timestamp = frame_number / fps
        output_path = os.path.splitext(video_path)[0] + ".jpg"

        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp),
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            output_path,
        ]
        subprocess.run(cmd, cwd=FFMPEG_BIN_DIR, capture_output=True, text=True, check=True)

        if log:
            log.emit("dlc", f"Extracted frame for {os.path.basename(video_path)}")

        if progress:
            progress.emit(round((index / total) * 100))

    return {
        "kind": "dlc_frames",
        "videos": len(request.video_paths),
    }


def check_dlc_folder_structure(config_path: str) -> tuple[bool, list[str]]:
    messages: list[str] = []

    if not config_path or not os.path.isfile(config_path):
        return False, ["Please select a valid config.yaml file."]

    try:
        _, usable_config_path, was_repaired = prepare_dlc_config(config_path)
        if was_repaired:
            messages.append("The selected config.yaml was broken and was repaired successfully.")
            messages.append(f"Using repaired config: {usable_config_path}")
    except Exception as exc:
        return False, [f"Config.yaml is invalid and could not be repaired: {exc}"]

    project_root = os.path.dirname(usable_config_path)
    required_folders = ["dlc-models", "evaluation-results", "labeled-data", "training-datasets", "videos"]
    required_files = ["config.yaml"]

    for folder in required_folders:
        folder_path = os.path.join(project_root, folder)
        if os.path.isdir(folder_path):
            messages.append(f"The folder {folder} is OK")
        else:
            messages.append(f"The folder '{folder}' is NOT present")
            return False, messages

    for file_name in required_files:
        file_path = os.path.join(project_root, file_name)
        if not os.path.isfile(file_path):
            messages.append(f"The project's {file_name} is NOT present")
            return False, messages

    messages.append("The folder structure is correct.")
    return True, messages


def _get_required_files_status(folder_path: Path, task_type: str) -> tuple[dict[str, bool], list[str]]:
    file_names = [path.name for path in folder_path.iterdir() if path.is_file()]
    lower_names = [name.lower() for name in file_names]

    status = {
        "hasFilteredCsv": any(name.endswith("filtered.csv") and not name.endswith("filtered_skeleton.csv") for name in lower_names),
        "hasSkeletonFilteredCsv": any(name.endswith("filtered_skeleton.csv") for name in lower_names),
        "hasImageFile": any(name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")) for name in lower_names),
        "hasRoiFile": any(name.endswith("roi.csv") for name in lower_names),
        "hasLeftRoiFile": any(name.endswith("roil.csv") for name in lower_names),
        "hasRightRoiFile": any(name.endswith("roir.csv") for name in lower_names),
    }

    missing_files: list[str] = []

    if not status["hasFilteredCsv"]:
        missing_files.append(" - filtered.csv")
    if not status["hasSkeletonFilteredCsv"]:
        missing_files.append(" - filtered_skeleton.csv")
    if not status["hasImageFile"]:
        missing_files.append(" - screenshot of the video")

    if task_type == "njr":
        if not status["hasLeftRoiFile"]:
            missing_files.append(" - roiL.csv")
        if not status["hasRightRoiFile"]:
            missing_files.append(" - roiR.csv")
    elif task_type == "social_recognition":
        if not status["hasRoiFile"]:
            missing_files.append(" - roi.csv")

    return status, missing_files


def _build_safe_destination(unwanted_folder: Path, source_path: Path) -> Path:
    destination = unwanted_folder / source_path.name
    if not destination.exists():
        return destination

    stem = source_path.stem if source_path.is_file() else source_path.name
    suffix = source_path.suffix if source_path.is_file() else ""
    counter = 1

    while True:
        candidate = unwanted_folder / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def run_clear_unused_files(request: DLCClearUnusedFilesRequest, progress=None, log=None, warning=None):
    folder_path = Path(request.folder_path)

    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"Invalid folder: {folder_path}")

    task_type = request.task_type.strip().lower()

    unwanted_folder = folder_path / "unwanted_files"
    unwanted_folder.mkdir(exist_ok=True)

    video_suffixes = VALID_VIDEO_EXTENSIONS
    image_suffixes = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    entry_list = [path for path in folder_path.iterdir()]
    moved_files: list[str] = []

    total = max(len(entry_list), 1)

    protected_folder_names = {"unwanted_files"}

    for index, entry_path in enumerate(entry_list, start=1):
        lower_name = entry_path.name.lower()
        keep_entry = False

        if entry_path.is_dir():
            if lower_name in protected_folder_names:
                keep_entry = True

            if not keep_entry:
                destination = _build_safe_destination(unwanted_folder, entry_path)
                shutil.move(str(entry_path), str(destination))
                moved_files.append(entry_path.name)
                if log:
                    log.emit("dlc", f"Moved folder {entry_path.name} to unwanted_files")

        else:
            suffix = entry_path.suffix.lower()

            if suffix in video_suffixes:
                keep_entry = True
            elif suffix in image_suffixes:
                keep_entry = True
            elif "roi" in lower_name and suffix == ".csv":
                keep_entry = True
            elif lower_name.endswith("filtered.csv"):
                keep_entry = True
            elif lower_name.endswith("filtered_skeleton.csv"):
                keep_entry = True

            if not keep_entry:
                destination = _build_safe_destination(unwanted_folder, entry_path)
                shutil.move(str(entry_path), str(destination))
                moved_files.append(entry_path.name)
                if log:
                    log.emit("dlc", f"Moved {entry_path.name} to unwanted_files")

        if progress:
            progress.emit(round((index / total) * 100))

    status, missing_files = _get_required_files_status(folder_path, task_type)

    if not missing_files:
        if log:
            log.emit("dlc", "All required files are present.")
    else:
        if log:
            log.emit("dlc", "There are missing files in the folder")

    return {
        "kind": "dlc_clear_unused_files",
        "folderPath": str(folder_path),
        "movedFiles": moved_files,
        "missingFiles": missing_files,
        "allRequiredPresent": len(missing_files) == 0,
    }
