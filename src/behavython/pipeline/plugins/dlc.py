from __future__ import annotations

import os
import sys
import shutil
import logging
import subprocess
from tqdm import tqdm
from pathlib import Path
from typing import Any
from behavython.core.defaults import ANALYSIS_REQUIRED_SUFFIXES, MAZE_EXPERIMENT_TYPES
from behavython.core.utils import load_or_repair_dlc_yaml, get_ffmpeg_path, get_ffprobe_path, detect_gpu
from behavython.core.exceptions import AnalysisError
from behavython.services.validation import validate_config_path, validate_video_paths
from behavython.services.logging import capture_external_output
from behavython.pipeline.models import (
    DLCClearUnusedFilesRequest,
    DLCFrameExtractionRequest,
    DLCVideoAnalysisRequest,
    DLCAnnotatedVideoRequest,
)

app_logger = logging.getLogger("behavython")
dlc_logger = logging.getLogger("behavython.dlc")
console_logger = logging.getLogger("behavython.console")


def load_deeplabcut():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("MPLBACKEND", "Agg")

    if "deeplabcut" not in sys.modules:
        with capture_external_output("behavython.external"):
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

    # Check if the user has a gpu available for DLC to use, and if not, warn them that the analysis may be very slow
    try:
        has_gpu, _ = detect_gpu()

        if has_gpu:
            dlc_logger.info("GPU detected and will be used by DeepLabCut.")
            gpu_to_use = 0
        else:
            if warning:
                warning.emit("Warning", "No GPU detected. DeepLabCut will run on CPU (this can be very slow).")
            dlc_logger.warning("No GPU detected. Falling back to CPU.")
            gpu_to_use = None

    except Exception as e:
        dlc_logger.warning(f"GPU detection failed: {e}")
        gpu_to_use = None

    if errors:
        raise AnalysisError("\n".join(errors))

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

    # Admittedly, not the best approach to log progress as I'm inserting
    # a little bit of overhead by analyzing videos one by one,
    # but deeplabcut doesn't provide a way to get progress when analyzing
    # multiple videos at once, and this way i can at least log the some
    # information about which video is being analyzed in the console output.
    console_logger.info(f"DLC analysis start | videos={len(request.video_paths)}")
    console_logger.info("DLC note | Progress bar may appear stuck during first video")
    console_logger.info("DLC note | Processing is approximately real-time for GPUs like the RTX 3060")
    console_logger.info("DLC note | It may be fast for newer GPUs and slower for older GPUs or CPU-only")
    for video in tqdm(request.video_paths, desc="Analyzing videos", unit="video"):
        # repair config for each video in case DLC moves or creates files during analysis that break the config.yaml
        # this is a bit of a band-aid for DLC's tendency to break the config.yaml, but it should help make the process more robust overall
        # Need to look into why DLC corrupts the config.yaml every time it analyzes a video
        # Also done below
        _, usable_config_path, was_repaired = prepare_dlc_config(request.config_path)
        with capture_external_output("behavython.external"):
            deeplabcut.analyze_videos(
                usable_config_path,
                [video],
                videotype=extension,
                shuffle=1,
                trainingsetindex=0,
                gputouse=gpu_to_use,
                allow_growth=True,
                save_as_csv=True,
            )

    if log:
        log.emit("dlc", "Filtering predictions...")
    dlc_logger.info("Calling deeplabcut.filterpredictions")

    if progress:
        progress.emit(60)

    console_logger.info(f"Filtering predictions with DeepLabCut for {len(request.video_paths)} video(s)")
    with capture_external_output("behavython.external"):
        _, usable_config_path, was_repaired = prepare_dlc_config(request.config_path)
        deeplabcut.filterpredictions(
            usable_config_path,
            request.video_paths,
            videotype=extension,
            shuffle=1,
            trainingsetindex=0,
            filtertype="median",
            save_as_csv=True,
        )
        deeplabcut.analyzeskeleton(
            usable_config_path,
            request.video_paths,
            shuffle=1,
            trainingsetindex=0,
            filtered=True,
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


def _get_video_fps(video_path: str) -> float:
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe,
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
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    numerator, denominator = map(float, result.stdout.strip().split("/"))
    return numerator / denominator


def _get_video_duration(video_path: str) -> float:
    ffprobe = get_ffprobe_path()
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def run_extract_frames(request: DLCFrameExtractionRequest, progress=None, log=None, warning=None):
    errors = validate_video_paths(request.video_paths)
    if errors:
        raise AnalysisError("\n".join(errors))

    total = len(request.video_paths)

    for index, video_path in enumerate(request.video_paths, start=1):
        fps = _get_video_fps(video_path)
        duration = _get_video_duration(video_path)

        frame_number = request.override_frame_number
        if frame_number is None:
            frame_number = int((duration * fps) * 0.5)

        timestamp = frame_number / fps
        output_path = os.path.splitext(video_path)[0] + ".jpg"
        ffmpeg = get_ffmpeg_path()

        cmd = [
            ffmpeg,
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
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        console_logger.info(f"Extracted frame for {os.path.basename(video_path)} at timestamp {timestamp:.2f}s (frame {frame_number})")

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


def _is_file_needed_for_task(file_path: Path, task_type: str) -> bool:
    lower_name = file_path.name.lower()
    suffix = file_path.suffix.lower()
    normalized_task = task_type.strip().lower().replace(" ", "_")

    if suffix in ANALYSIS_REQUIRED_SUFFIXES["video"]:
        return True
    if suffix in ANALYSIS_REQUIRED_SUFFIXES["image"]:
        return True
    if lower_name.endswith(ANALYSIS_REQUIRED_SUFFIXES["position"]) and not lower_name.endswith(ANALYSIS_REQUIRED_SUFFIXES["skeleton"]):
        return True
    if lower_name.endswith(ANALYSIS_REQUIRED_SUFFIXES["skeleton"]):
        return True
    if normalized_task in {"social_discrimination", "njr"}:
        if any(lower_name.endswith(ending) for ending in ANALYSIS_REQUIRED_SUFFIXES["social_discrimination_rois"]):
            return True
    elif normalized_task == "social_recognition":
        if any(lower_name.endswith(ending) for ending in ANALYSIS_REQUIRED_SUFFIXES["social_recognition_rois"]):
            return True
    elif normalized_task in MAZE_EXPERIMENT_TYPES:
        if suffix in ANALYSIS_REQUIRED_SUFFIXES["maze_rois"]:
            return True

    return False


def _get_required_files_status(folder_path: Path, task_type: str) -> tuple[dict[str, bool], list[str]]:
    file_names = [path.name for path in folder_path.iterdir() if path.is_file()]
    lower_names = [name.lower() for name in file_names]

    normalized_task = task_type.strip().lower().replace(" ", "_")
    missing_files: list[str] = []

    status = {
        "has_video": any(name.endswith(ANALYSIS_REQUIRED_SUFFIXES["video"]) for name in lower_names),
        "has_filtered_csv": any(
            name.endswith(ANALYSIS_REQUIRED_SUFFIXES["position"]) and not name.endswith(ANALYSIS_REQUIRED_SUFFIXES["skeleton"]) for name in lower_names
        ),
        "has_skeleton_filtered_csv": any(name.endswith(ANALYSIS_REQUIRED_SUFFIXES["skeleton"]) for name in lower_names),
        "has_image_file": any(name.endswith(ANALYSIS_REQUIRED_SUFFIXES["image"]) for name in lower_names),
        "has_roi_file": any(name.endswith(ending) for name in lower_names for ending in ANALYSIS_REQUIRED_SUFFIXES["social_recognition_rois"]),
        "has_left_roi_file": any(name.endswith("roil.csv") for name in lower_names),
        "has_right_roi_file": any(name.endswith("roir.csv") for name in lower_names),
        "has_geometry_json": any(name.endswith(ANALYSIS_REQUIRED_SUFFIXES["maze_rois"]) for name in lower_names),
    }

    if not status["has_filtered_csv"]:
        missing_files.append(" - filtered.csv")
    if not status["has_skeleton_filtered_csv"]:
        missing_files.append(" - filtered_skeleton.csv")
    if not status["has_image_file"]:
        missing_files.append(" - screenshot of the video")
    if normalized_task in {"social_discrimination", "njr"}:
        if not status["has_left_roi_file"]:
            missing_files.append(" - roiL.csv")
        if not status["has_right_roi_file"]:
            missing_files.append(" - roiR.csv")
    elif normalized_task == "social_recognition":
        if not status["has_roi_file"]:
            missing_files.append(" - roi.csv")
    elif normalized_task in MAZE_EXPERIMENT_TYPES:
        if not status["has_geometry_json"]:
            missing_files.append(" - arena geometry (.json)")

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
        raise AnalysisError(f"Invalid folder: {folder_path}")

    task_type = request.task_type.strip().lower().replace(" ", "_")

    unwanted_folder = folder_path / "unwanted_files"
    unwanted_folder.mkdir(exist_ok=True)

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
            keep_entry = _is_file_needed_for_task(entry_path, task_type)

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

    if console_logger:
        console_logger.info(f"Cleared unused files for DLC folder: {folder_path}")
        console_logger.info(f"Moved {len(moved_files)} file(s) to {unwanted_folder}")
        if missing_files:
            formatted_missing = "\n".join(missing_files)  # Fixed join string format
            console_logger.warning(f"Missing required files:\n{formatted_missing}")

    return {
        "kind": "dlc_clear_unused_files",
        "folder_path": str(folder_path),
        "moved_files": moved_files,
        "missing_files": missing_files,
        "all_required_present": len(missing_files) == 0,
    }


def run_create_annotated_video(
    request: DLCAnnotatedVideoRequest,
    progress=None,
    log=None,
    warning=None,
) -> dict:
    """
    Creates an annotated video using DeepLabCut.
    Handles uncleaned folders and utilizes automated YAML repair.
    """
    if log:
        log.emit("dlc", "Importing DeepLabCut...")
    try:
        import deeplabcut
    except ImportError:
        log.emit("dlc", "[ERROR]: DeepLabCut not found in environment.")
        return {"success": False, "error": "ImportError"}

    log.emit("dlc", "Validating DLC configuration...")
    try:
        config_data, config_path_used, was_repaired = load_or_repair_dlc_yaml(request.config_path)
        if was_repaired:
            log.emit("dlc", f"Config repaired: {os.path.basename(config_path_used)}")
    except Exception as e:
        log.emit("dlc", f"[ERROR]: Could not load config: {str(e)}")
        return {"success": False, "error": str(e)}

    video_suffixes = {Path(video_path).suffix for video_path in request.video_paths if Path(video_path).suffix}
    if len(video_suffixes) == 1:
        videotype = video_suffixes.pop()
    else:
        # Fallback to .mp4 to preserve previous behavior when multiple or no suffixes are found
        videotype = ".mp4"

    # DLC needs the .h5 and .pickle files in the same directory as the video
    for video_path in request.video_paths:
        video_dir = os.path.dirname(video_path)
        unwanted_dir = os.path.join(video_dir, "unwanted_files")
        required_suffixes = ("_filtered.h5", "_meta.pickle")

        if os.path.exists(unwanted_dir):
            files_in_unwanted = [f for f in os.listdir(unwanted_dir) if f.endswith(required_suffixes)]
            for f in files_in_unwanted:
                src = os.path.join(unwanted_dir, f)
                dst = os.path.join(video_dir, f)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
                    log.emit("dlc", f"Restored from unwanted_files: {f}")

    log.emit("dlc", f"Starting labeled video generation for {len(request.video_paths)} videos...")
    console_logger.info(f"Creating annotated videos with DeepLabCut for {len(request.video_paths)} video(s)")

    try:
        deeplabcut.create_labeled_video(
            str(config_path_used), request.video_paths, videotype=videotype, filtered=True, draw_skeleton=True, destfolder=request.output_path
        )
    except AnalysisError as e:
        dlc_logger.exception("DLC labeled video creation failed")
        log.emit("dlc", f"[ERROR]: DLC error: {str(e)}")
        raise AnalysisError(f"[ERROR]: DLC error: {str(e)}")

    return {
        "kind": "dlc_annotated_video",
        "videos": len(request.video_paths),
        "output_path": request.output_path,
        "config_was_repaired": was_repaired,
        "config_path_used": str(config_path_used),
    }
