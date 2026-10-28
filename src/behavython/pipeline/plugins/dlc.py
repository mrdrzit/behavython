from __future__ import annotations

import os
import re
import sys
import shutil
import logging
import subprocess
import tempfile
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Any
from behavython.core.defaults import (
    ANALYSIS_REQUIRED_SUFFIXES,
    MAZE_EXPERIMENT_TYPES,
    ROI_NETWORK_BODYPARTS,
    ROI_FRAMES_PER_VIDEO,
    ROI_SKIP_START_FRACTION,
    ROI_FRAME_DEVIATION_WARNING_PX,
    ROI_DIAMETER_OUTLIER_FRACTION,
    ROI_CHECK_FOLDER,
    CV2_ROI_CHECK_COLOR,
    CV2_ROI_POINT_COLOR,
)
from behavython.core.utils import load_or_repair_dlc_yaml, get_ffmpeg_path, get_ffprobe_path, detect_gpu
from behavython.core.exceptions import AnalysisError
from behavython.services.validation import validate_config_path, validate_video_paths, validate_dlc_backend
from behavython.services.logging import capture_external_output
from behavython.pipeline.models import (
    DLCClearUnusedFilesRequest,
    DLCCreateRoisRequest,
    DLCFrameExtractionRequest,
    DLCVideoAnalysisRequest,
    DLCAnnotatedVideoRequest,
    DLCAnalyzeFramesRequest,
    DLCLikelihoodPlotRequest,
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


def snapshot_index_from_file(snapshot_path: str | Path) -> tuple[int, str]:
    """Index DLC uses for a snapshot file, and the snapshot name.

    PyTorch snapshots are ``snapshot-*.pt``, ordered by epoch with the "best" snapshot last.
    TensorFlow snapshots are ``snapshot-<iteration>.index/.meta/.data-*``, ordered by iteration.
    Raises ValueError for files that are not regular pose snapshots.
    """
    path = Path(snapshot_path)
    folder = path.parent

    if path.suffix == ".pt":
        snapshots = [p for p in folder.glob("snapshot-*.pt") if "detector" not in p.name]

        def sort_key(p: Path) -> tuple[bool, int]:
            digits = "".join(c for c in p.stem if c.isdigit())
            return ("best" in p.stem, int(digits) if digits else 0)

        names = [p.name for p in sorted(snapshots, key=sort_key)]
        if path.name not in names:
            raise ValueError("Please select a regular (non-detector) snapshot-*.pt file.")
        return names.index(path.name), path.name

    # TensorFlow: any of the checkpoint files names the snapshot; DLC lists them through the .index files.
    name = path.name.split(".")[0]
    iterations = sorted(int(p.name.split(".")[0].split("-")[1]) for p in folder.glob("snapshot-*.index"))
    try:
        return iterations.index(int(name.split("-")[1])), name
    except (IndexError, ValueError):
        raise ValueError("Please select a snapshot-*.pt (PyTorch) or snapshot-*.index (TensorFlow) file.") from None


def prepare_snapshot_config(config_path: str, config_dict: dict[str, Any], snapshot_index: int | None) -> tuple[str, dict[str, Any]]:
    """Config path and extra DLC kwargs that make DLC use ``snapshot_index``.

    PyTorch takes ``snapshot_index`` as an argument. TensorFlow only reads ``snapshotindex`` from config.yaml,
    so it gets a ``*_snapshot.yaml`` copy next to the original with that key set. The original is not touched.
    """
    if snapshot_index is None:
        return config_path, {}
    if validate_dlc_backend(config_dict, config_path) == "pytorch":
        return config_path, {"snapshot_index": snapshot_index}

    source = Path(config_path)
    text = source.read_text(encoding="utf-8")
    line = f"snapshotindex: {snapshot_index}"
    text, count = re.subn(r"^snapshotindex:.*$", line, text, flags=re.MULTILINE)
    if count == 0:
        text = text.rstrip("\n") + f"\n{line}\n"
    out_path = source.with_name(f"{source.stem}_snapshot{source.suffix}")
    out_path.write_text(text, encoding="utf-8")
    return str(out_path), {}


def _emit_config_repair_logs(original_path: str, usable_path: str, was_repaired: bool, log) -> None:
    if not log:
        return

    if was_repaired:
        log.emit("dlc", "Broken config detected and repaired.")
        log.emit("dlc", f"Original config: {original_path}")
        log.emit("dlc", f"Using repaired config: {usable_path}")
    else:
        log.emit("dlc", "Config.yaml validated successfully.")


def get_dlc_file_status(video_path: str) -> dict:
    video = Path(video_path)
    stem = video.stem
    folder = video.parent

    has_base = False
    has_filtered = False

    if not folder.exists() or not folder.is_dir():
        return {"needs_analysis": True, "needs_filtering": True}

    for file in folder.iterdir():
        if not file.is_file():
            continue

        if not file.name.startswith(stem):
            continue

        remaining = file.name[len(stem) :]
        if not remaining.startswith(("DLC", "SuperAnimal", "CollectedData")):
            continue

        # Ignore known non-dlc tracking files
        if file.name.endswith(("_roi.csv", "roil.csv", "roir.csv", ".jpg", ".mp4", ".avi", ".mov", ".pickle", ".yaml", ".json")):
            continue

        if file.suffix in [".h5", ".csv"]:
            if "filtered" in file.name:
                has_filtered = True
            else:
                has_base = True

    needs_analysis = not has_base and not has_filtered
    needs_filtering = not has_filtered

    return {"needs_analysis": needs_analysis, "needs_filtering": needs_filtering}


def infer_dlc_shuffle_and_trainingsetindex(config_path: str, config_dict: dict) -> tuple[int, int]:
    """
    Attempts to infer the optimal shuffle and trainingsetindex by inspecting
    the dlc-models folder for the current iteration.
    Falls back to (1, 0) if inference fails.
    """
    import re

    project_path = Path(config_path).parent
    iteration = config_dict.get("iteration", 0)
    dlc_models = project_path / "dlc-models" / f"iteration-{iteration}"

    default_shuffle, default_trainingsetindex = 1, 0

    if not dlc_models.exists() or not dlc_models.is_dir():
        return default_shuffle, default_trainingsetindex

    pattern = re.compile(r"trainset(\d+)shuffle(\d+)")

    found = []
    for d in dlc_models.iterdir():
        if d.is_dir():
            match = pattern.search(d.name)
            if match:
                fraction_pct, shuffle = int(match.group(1)), int(match.group(2))
                found.append((fraction_pct, shuffle))

    if not found:
        return default_shuffle, default_trainingsetindex

    found.sort(key=lambda x: (x[1], x[0]), reverse=True)
    fraction_pct, shuffle = found[0]

    training_fractions = config_dict.get("TrainingFraction", [0.95])
    trainingsetindex = 0
    for i, frac in enumerate(training_fractions):
        if int(frac * 100) == fraction_pct:
            trainingsetindex = i
            break

    return shuffle, trainingsetindex


def run_dlc_video_analysis(request: DLCVideoAnalysisRequest, progress=None, log=None, warning=None):
    errors = validate_config_path(request.config_path) + validate_video_paths(request.video_paths)

    if errors:
        raise AnalysisError("\n".join(errors))

    config_dict, usable_config_path, was_repaired = prepare_dlc_config(request.config_path)
    _emit_config_repair_logs(
        request.config_path,
        usable_config_path,
        was_repaired,
        log,
    )

    backend = validate_dlc_backend(config_dict)

    dlc_logger.info("DeepLabCut network backend: %s", backend)

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

    videos_to_analyze = []
    videos_to_filter = []

    for video in request.video_paths:
        status = get_dlc_file_status(video)
        if status["needs_analysis"]:
            videos_to_analyze.append(video)
            videos_to_filter.append(video)
        elif status["needs_filtering"]:
            videos_to_filter.append(video)
        else:
            console_logger.info(f"Skipping {Path(video).name}: Already analyzed and filtered.")

    if not videos_to_analyze and not videos_to_filter:
        console_logger.info("All videos are already fully processed. Skipping DLC analysis.")
        if progress:
            progress.emit(100)
        return {
            "kind": "dlc_analysis",
            "videos": len(request.video_paths),
            "config_path_used": usable_config_path,
            "config_was_repaired": was_repaired,
        }

    deeplabcut = load_deeplabcut()

    if log:
        log.emit("dlc", f"Using DeepLabCut version {deeplabcut.__version__}")

    extension = os.path.splitext(request.video_paths[0])[1].lower()

    dlc_config_path, snapshot_kwargs = prepare_snapshot_config(usable_config_path, config_dict, request.snapshot_index)

    if videos_to_analyze:
        if log:
            log.emit("dlc", "Analyzing videos...")
        dlc_logger.info("Calling deeplabcut.analyze_videos")

        if progress:
            progress.emit(40)

        console_logger.info(f"DLC analysis start | videos={len(videos_to_analyze)}")
        for i, video in enumerate(videos_to_analyze, 1):
            console_logger.info(f"Processing video {i}/{len(videos_to_analyze)}: {Path(video).name}")
            _, usable_config_path, was_repaired = prepare_dlc_config(request.config_path)
            with capture_external_output("behavython.external"):
                deeplabcut.analyze_videos(
                    dlc_config_path,
                    [video],
                    videotype=extension,
                    gputouse=gpu_to_use,
                    save_as_csv=True,
                    **snapshot_kwargs,
                )

    if videos_to_filter:
        if log:
            log.emit("dlc", "Filtering predictions...")

        if progress:
            progress.emit(60)

        console_logger.info(f"Filtering predictions with DeepLabCut for {len(videos_to_filter)} video(s)")
        with capture_external_output("behavython.external"):
            _, usable_config_path, was_repaired = prepare_dlc_config(request.config_path)
            deeplabcut.filterpredictions(
                dlc_config_path,
                videos_to_filter,
                videotype=extension,
                filtertype="median",
                save_as_csv=True,
                **snapshot_kwargs,
            )
            try:
                deeplabcut.analyzeskeleton(
                    dlc_config_path,
                    videos_to_filter,
                    filtered=True,
                    save_as_csv=True,
                    **snapshot_kwargs,
                )
            except Exception as e:
                if "skeleton" in str(e).lower():
                    msg = "No skeleton defined in config.yaml. Skipping skeleton analysis."
                    if warning:
                        warning.emit("Warning", msg)
                else:
                    raise

    if request.create_plots:
        if progress:
            progress.emit(80)
        if log:
            log.emit("dlc", "Generating likelihood plots...")

        from behavython.pipeline.plotting import plot_custom_likelihood

        scorer = _scorer_for_snapshot(dlc_config_path, request.snapshot_index)
        for video in videos_to_filter:
            video_path = Path(video)
            candidates = [f for f in video_path.parent.glob(f"{video_path.stem}*_filtered.h5") if scorer is None or scorer in f.name]
            if not candidates:
                dlc_logger.warning("No filtered .h5 found for %s; skipping likelihood plot.", video_path.name)
                continue
            h5_file = max(candidates, key=lambda f: f.stat().st_mtime)
            try:
                plot_custom_likelihood(str(h5_file), config_dict, str(h5_file.with_name(h5_file.stem + "_likelihood.png")))
            except Exception as e:
                dlc_logger.error(f"Failed to plot {h5_file.name}: {e}")
                if warning:
                    warning.emit("Warning", f"Failed to plot likelihood for {h5_file.name}: {e}")

    if progress:
        progress.emit(100)

    dlc_logger.info("DLC analysis finished successfully")

    return {
        "kind": "dlc_analysis",
        "videos": len(request.video_paths),
        "config_path_used": usable_config_path,
        "config_was_repaired": was_repaired,
    }


def _scorer_for_snapshot(config_path: str, snapshot_index: int | None) -> str | None:
    """DLC scorer name (embedded in output file names) for a snapshot, or None if no snapshot was chosen."""
    if snapshot_index is None:
        return None
    from deeplabcut.utils import auxiliaryfunctions

    cfg = auxiliaryfunctions.read_config(config_path)
    # TensorFlow reads the index from the config; DLC 2.3 get_scorer_name has no snapshot_index argument.
    cfg["snapshotindex"] = snapshot_index
    kwargs = {"snapshot_index": snapshot_index} if validate_dlc_backend(cfg, config_path) == "pytorch" else {}
    scorer, _ = auxiliaryfunctions.get_scorer_name(cfg, 1, trainFraction=cfg["TrainingFraction"][0], **kwargs)
    return scorer


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


def _sample_frame_indices(frame_count: int) -> list[int]:
    start = frame_count * ROI_SKIP_START_FRACTION
    step = (frame_count - start) / ROI_FRAMES_PER_VIDEO
    return [int(start + step * (i + 0.5)) for i in range(ROI_FRAMES_PER_VIDEO)]


def predict_cup_points(config_path: str, video_paths: list[str], frames_dir: Path) -> dict[str, list[tuple[int, Path, np.ndarray]]]:
    """Runs the ROI network on frames sampled from each video.

    Returns, per video, a list of (frame index, frame image, points) where points has shape (4, 2)
    in ROI_NETWORK_BODYPARTS order. The frame images are written to ``frames_dir``.
    """
    config_dict, usable_config_path, _ = prepare_dlc_config(config_path)
    missing = [bp for bp in ROI_NETWORK_BODYPARTS if bp not in (config_dict.get("bodyparts") or [])]
    if missing:
        raise AnalysisError(f"The ROI network must have the bodyparts {', '.join(ROI_NETWORK_BODYPARTS)}. Missing: {', '.join(missing)}.")

    sources: dict[str, tuple[str, int]] = {}
    for video in video_paths:
        cap = cv2.VideoCapture(video)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in _sample_frame_indices(frame_count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                dlc_logger.warning("Could not read frame %d of %s", index, video)
                continue
            # Numbered names keep video names with spaces or accents out of OpenCV and DLC paths.
            name = f"{len(sources):05d}.png"
            cv2.imwrite(str(frames_dir / name), frame)
            sources[name] = (video, index)
        cap.release()

    if not sources:
        raise AnalysisError("No frames could be read from the selected videos.")

    try:
        has_gpu, _ = detect_gpu()
        gpu_to_use = 0 if has_gpu else None
    except Exception:
        gpu_to_use = None

    deeplabcut = load_deeplabcut()
    with capture_external_output("behavython.external"):
        deeplabcut.analyze_time_lapse_frames(usable_config_path, str(frames_dir), frametype=".png", gputouse=gpu_to_use)

    h5_files = list(frames_dir.glob("*.h5"))
    if not h5_files:
        raise AnalysisError("DeepLabCut did not produce predictions for the ROI frames.")
    predictions = pd.read_hdf(h5_files[0])
    # Shape (frames, 4, 2); selecting by level name works with or without an "individuals" level.
    coords = np.stack(
        [
            [predictions.xs((bp, axis), level=("bodyparts", "coords"), axis=1).iloc[:, 0].to_numpy(dtype=float) for axis in ("x", "y")]
            for bp in ROI_NETWORK_BODYPARTS
        ]
    ).transpose(2, 0, 1)

    results: dict[str, list[tuple[int, Path, np.ndarray]]] = {}
    for row_name, points in zip(predictions.index, coords):
        name = Path(row_name[-1] if isinstance(row_name, tuple) else str(row_name)).name
        video, index = sources[name]
        results.setdefault(video, []).append((index, frames_dir / name, points))
    return results


def _circle_from_points(points: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Center, width (left-right) and height (top-bottom) of the cup from the top, right, bottom and left points.

    A tilted cup projects as an ellipse, so the two axes are kept apart; the analysis uses their mean as the diameter.
    Each half-axis is truncated to whole pixels, as in the legacy ROI files.
    """
    center = points.mean(axis=0)
    width = int(np.linalg.norm(points[3] - points[1]) / 2) * 2
    height = int(np.linalg.norm(points[2] - points[0]) / 2) * 2
    return center, width, height


def run_create_rois(request: DLCCreateRoisRequest, progress=None, log=None, warning=None) -> dict:
    """Creates <video>_roi.csv for each video with the find_cup_circle ROI network.

    Existing _roi.csv files are never overwritten. A check image per video goes to ROI_CHECK_FOLDER.
    """
    errors = validate_config_path(request.config_path) + validate_video_paths(request.video_paths)
    if errors:
        raise AnalysisError("\n".join(errors))

    def report(message: str) -> None:
        console_logger.info(message)
        if log:
            log.emit("dlc", message)

    def flag(video: str, message: str) -> None:
        console_logger.warning(f"{Path(video).name}: {message}")
        if log:
            log.emit("dlc", f"WARNING {Path(video).name}: {message}")
        flagged.setdefault(Path(video).name, []).append(message)

    flagged: dict[str, list[str]] = {}
    skipped = []
    pending = []
    for video in request.video_paths:
        roi_path = Path(video).with_name(f"{Path(video).stem}_roi.csv")
        if roi_path.exists():
            skipped.append(video)
            report(f"Skipped {Path(video).name}: {roi_path.name} already exists.")
        else:
            pending.append(video)

    created: dict[str, float] = {}
    if pending:
        report(f"Predicting cup circles on {ROI_FRAMES_PER_VIDEO} frames from each of {len(pending)} video(s)...")
        if progress:
            progress.emit(10)

        with tempfile.TemporaryDirectory(prefix="behavython_roi_") as tmp:
            predictions = predict_cup_points(request.config_path, pending, Path(tmp))
            if progress:
                progress.emit(80)

            for video in pending:
                frames = predictions.get(video)
                if not frames:
                    flag(video, "no frames could be read, no ROI was created.")
                    continue

                median_points = np.median(np.stack([points for _, _, points in frames]), axis=0)
                center, width, height = _circle_from_points(median_points)
                diameter = (width + height) / 2

                for index, _, points in frames:
                    frame_center, frame_width, frame_height = _circle_from_points(points)
                    frame_diameter = (frame_width + frame_height) / 2
                    shift = float(np.linalg.norm(frame_center - center))
                    if shift > ROI_FRAME_DEVIATION_WARNING_PX or abs(frame_diameter - diameter) > ROI_FRAME_DEVIATION_WARNING_PX:
                        flag(
                            video,
                            f"frame {index}: circle is {shift:.0f} px from the median center, diameter {frame_diameter:.0f} vs {diameter:.0f} px.",
                        )

                roi_path = Path(video).with_name(f"{Path(video).stem}_roi.csv")
                roi = {"X": center[0], "Y": center[1], "BX": int(center[0]), "BY": int(center[1]), "Width": width, "Height": height}
                pd.DataFrame(roi, index=[0]).to_csv(roi_path, index=False)
                created[video] = diameter

                check_dir = Path(video).parent / ROI_CHECK_FOLDER
                check_dir.mkdir(exist_ok=True)
                image = cv2.imread(str(frames[len(frames) // 2][1]))
                cv2.ellipse(image, (int(center[0]), int(center[1])), (width // 2, height // 2), 0, 0, 360, CV2_ROI_CHECK_COLOR, 2)
                for x, y in median_points:
                    cv2.circle(image, (int(x), int(y)), 4, CV2_ROI_POINT_COLOR, -1)
                cv2.imwrite(str(check_dir / f"{Path(video).stem}_roi_check.jpg"), image)
                report(f"Created {roi_path.name} (center {center[0]:.1f}, {center[1]:.1f}; width {width} x height {height} px)")

    # A video whose circle is consistent across frames can still be wrong as a whole; compare it with the others.
    if len(created) >= 3:
        folder_median = float(np.median(list(created.values())))
        for video, diameter in created.items():
            if abs(diameter - folder_median) > ROI_DIAMETER_OUTLIER_FRACTION * folder_median:
                flag(video, f"diameter {diameter:.0f} px is far from the median of all videos ({folder_median:.0f} px).")

    if flagged:
        console_logger.warning(f"Check these ROIs in {ROI_CHECK_FOLDER}/ and redo them manually if needed: {', '.join(flagged)}")

    if progress:
        progress.emit(100)

    return {
        "kind": "dlc_create_rois",
        "created": len(created),
        "skipped": len(skipped),
        "flagged": list(flagged),
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
    elif normalized_task in MAZE_EXPERIMENT_TYPES and suffix in ANALYSIS_REQUIRED_SUFFIXES["maze_rois"]:
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
            name.endswith(ANALYSIS_REQUIRED_SUFFIXES["position"]) and not name.endswith(ANALYSIS_REQUIRED_SUFFIXES["skeleton"])
            for name in lower_names
        ),
        "has_skeleton_filtered_csv": any(name.endswith(ANALYSIS_REQUIRED_SUFFIXES["skeleton"]) for name in lower_names),
        "has_image_file": any(name.endswith(ANALYSIS_REQUIRED_SUFFIXES["image"]) for name in lower_names),
        "has_roi_file": any(
            name.endswith(ending) for name in lower_names for ending in ANALYSIS_REQUIRED_SUFFIXES["social_recognition_rois"]
        ),
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

    _status, missing_files = _get_required_files_status(folder_path, task_type)

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


def run_create_annotated_video(request: DLCAnnotatedVideoRequest, progress=None, log=None, warning=None) -> dict:
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
        log.emit("dlc", f"[ERROR]: Could not load config: {e!s}")
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

    dlc_config_path, snapshot_kwargs = prepare_snapshot_config(str(config_path_used), config_data, request.snapshot_index)
    try:
        deeplabcut.create_labeled_video(
            dlc_config_path,
            request.video_paths,
            videotype=videotype,
            filtered=True,
            draw_skeleton=True,
            destfolder=request.output_path,
            **snapshot_kwargs,
        )
    except AnalysisError as e:
        dlc_logger.exception("DLC labeled video creation failed")
        log.emit("dlc", f"[ERROR]: DLC error: {e!s}")
        raise AnalysisError(f"[ERROR]: DLC error: {e!s}")

    return {
        "kind": "dlc_annotated_video",
        "videos": len(request.video_paths),
        "output_path": request.output_path,
        "config_was_repaired": was_repaired,
        "config_path_used": str(config_path_used),
    }


def run_analyze_frames(request: DLCAnalyzeFramesRequest, progress=None, log=None, warning=None) -> dict:
    """
    Orchestrates the Model-Assisted Label Transfer workflow.
    Scans the DLC project, resolves frame assets, and runs DLC inference.
    The session handles backups and rollback automatically.
    """
    from behavython.services.dlc_assisted_label_session import DLCAssistedLabelSession

    if log:
        console_logger.info("Starting Assisted Label session...")

    if progress:
        progress.emit(5)

    try:
        with DLCAssistedLabelSession(request) as session:
            if log:
                log.emit(
                    "dlc",
                    f"Model: {session.model_metadata.get('network_name', 'unknown')} | Scorer: {session.model_metadata.get('scorer', 'unknown')}",
                )
                console_logger.info(
                    f"Model: {session.model_metadata.get('network_name', 'unknown')} | Scorer: {session.model_metadata.get('scorer', 'unknown')}"
                )

            if progress:
                progress.emit(15)

            if log:
                log.emit("dlc", "Resolving frame assets...")
                console_logger.info("Resolving frame assets...")
            assets = session.resolve_frame_assets()

            for warn_msg in assets.warnings:
                dlc_logger.warning(warn_msg)
                if warning:
                    warning.emit("Warning", warn_msg)
                console_logger.warning(warn_msg)

            if not assets.is_ready:
                raise AnalysisError(f"No frames found in: {request.frames_folder}\nAdd frames or enable frame extraction in the request.")

            # Calculate total frames for logging
            total_frames = sum(len(list(f.glob("*"))) for f in assets.video_folders)

            if log:
                log.emit("dlc", f"{len(assets.video_folders)} folder(s) ({total_frames} frames) ready for inference")
                if assets.newly_extracted:
                    log.emit("dlc", f"{len(assets.newly_extracted)} frame(s) extracted from video(s)")

            if progress:
                progress.emit(35)

            if log:
                log.emit("dlc", "Running DLC inference (this may take a while)...")
                console_logger.info("Running DLC inference (this may take a while)...")
            generated_files = session.run_inference(assets)

            if log:
                log.emit("dlc", f"Inference complete | {len(generated_files)} video folder(s) processed")
                console_logger.info(f"Inference complete | {len(generated_files)} video folder(s) processed")

            # 4. Verification Plots
            if generated_files:
                # Merge all standardized H5s for a unified verification report
                all_dfs = [pd.read_hdf(f) for f in generated_files]
                if all_dfs:
                    merged_df = pd.concat(all_dfs)
                    merged_path = session.session_temp / "merged_labels_for_verification.h5"
                    merged_df.to_hdf(merged_path, key="df_with_missing", mode="w")
                    session.generate_verification_plots(merged_path)

            if progress:
                progress.emit(90)

            session.commit()
            console_logger.info("Assisted labeling session committed successfully")

    except AnalysisError:
        raise
    except Exception as e:
        dlc_logger.exception("Assisted Label session failed unexpectedly")
        raise AnalysisError(f"Assisted labeling failed: {e}") from e

    if progress:
        progress.emit(100)

    if log:
        log.emit("dlc", "Assisted Label session completed successfully.")

    return {
        "kind": "dlc_analyze_frames",
        "folders_processed": len(assets.video_folders),
        "frames_extracted": len(assets.newly_extracted),
        "standardized_labels_created": len(generated_files),
        "scorer": session.model_metadata.get("scorer", ""),
        "target_folder": str(session.target_folder),
    }


def run_generate_likelihood_plots(request: DLCLikelihoodPlotRequest, progress=None, log=None, warning=None) -> dict:
    from behavython.pipeline.plotting import plot_custom_likelihood

    if log:
        log.emit("dlc", "Starting custom likelihood plot generation...")

    config_dict, usable_config_path, _was_repaired = prepare_dlc_config(request.config_path)

    folder = Path(request.folder_path)
    if not folder.exists() or not folder.is_dir():
        raise AnalysisError(f"Invalid folder: {folder}")

    # Find all .h5 files
    h5_files = list(folder.glob("*_filtered.h5")) + list(folder.glob("CollectedData_*.h5"))
    unwanted = folder / "unwanted_files"
    if unwanted.exists() and unwanted.is_dir():
        h5_files.extend(unwanted.glob("*_filtered.h5"))
        h5_files.extend(unwanted.glob("CollectedData_*.h5"))
    h5_files = list(set(h5_files))  # Remove duplicates if any overlap

    scorer = _scorer_for_snapshot(usable_config_path, request.snapshot_index)
    if scorer is not None:
        h5_files = [f for f in h5_files if f.name.startswith("CollectedData_") or scorer in f.name]

    if not h5_files:
        raise AnalysisError(f"No DLC tracking data (.h5) found in {folder.name}.")

    total = len(h5_files)
    generated_count = 0

    if progress:
        progress.emit(10)

    console_logger.info(f"Generating custom likelihood plots for {total} files in {folder.name}")

    for idx, h5_file in enumerate(h5_files, start=1):
        try:
            output_name = h5_file.stem + "_likelihood.png"
            output_path = folder / output_name

            plot_custom_likelihood(str(h5_file), config_dict, str(output_path))
            generated_count += 1

            if log:
                log.emit("dlc", f"Generated: {output_name}")

        except Exception as e:
            dlc_logger.error(f"Failed to plot {h5_file.name}: {e}")
            if warning:
                warning.emit("Warning", f"Failed to plot likelihood for {h5_file.name}: {e}")

        if progress:
            progress.emit(10 + int(90 * (idx / total)))

    console_logger.info(f"Finished generating {generated_count} likelihood plot(s).")

    if progress:
        progress.emit(100)

    return {
        "kind": "dlc_likelihood_plots",
        "plots_generated": generated_count,
        "folder": str(folder),
    }
