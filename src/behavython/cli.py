"""
behavython-cli — Headless command-line entry point for the Behavython analysis pipeline.

This module is completely isolated from PySide6 and Qt.
It constructs an AnalysisRequest from a JSON config file and delegates directly
to run_analysis_workflow(), the same function the GUI uses internally.

Usage:
    behavython-cli --config run.json [--files file1.csv ...] [--output /path/to/out] [--no-plots]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from behavython.services.logging import AppLoggingService

from behavython.core.defaults import (
    EXPERIMENT_TYPES,
    MAZE_EXPERIMENT_TYPES,
    ANALYSIS_REQUIRED_SUFFIXES,
)
from behavython.core.paths import PACKAGE_ROOT


# ===========================================================================
# Section: Constants & Configuration
# ===========================================================================

_VALID_EXPERIMENT_TYPES: frozenset[str] = frozenset(EXPERIMENT_TYPES) | MAZE_EXPERIMENT_TYPES

_VALID_FIG_RESOLUTIONS = {
    "640x480": ["640", "480"],
    "1280x720": ["1280", "720"],
    "1920x1080": ["1920", "1080"],
    "2560x1440": ["2560", "1440"],
}

_ANIMAL_THRESHOLD = {
    "mouse": 0.0267,
    "rat": 0.0667,
}

_ANALYSIS_SUFFIXES: frozenset[str] = frozenset(
    Path(s).suffix if not s.startswith(".") else s for values in ANALYSIS_REQUIRED_SUFFIXES.values() for s in values
)

_DLC_ACTIONS = frozenset(["run_tracking", "likelihood_plots", "annotate_video", "transfer_labels"])

_INPUTLESS_ACTIONS = frozenset({"init", "transfer_labels"})

_TERMINAL_ACTIONS: tuple[str, ...] = (
    "init",
    "transfer_labels",
    "standardize",
    "run_cropping",
    "extract_frames",
    "cleanup",
    "run_tracking",
    "likelihood_plots",
    "annotate_video",
)


# ===========================================================================
# Section: CLI Adapters
# ===========================================================================

class _ProgressAdapter:
    """Mimics a Qt Signal with a single .emit(value: int) method."""

    def emit(self, value: int) -> None:
        """Silenced in CLI to avoid overlap with standard log messages."""
        pass


class _LogAdapter:
    """Mimics a Qt Signal with a .emit(kind: str, message: str) method."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, kind: str, message: str) -> None:
        self._logger.info("[%s] %s", kind.upper(), message)


class _WarningAdapter:
    """Mimics a Qt Signal with a .emit(title: str, message: str) method."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, title: str, message: str) -> None:
        msg = f"{title}: {message}" if title.lower() not in message.lower() else message
        self._logger.warning(msg)


# ===========================================================================
# Section: Data Models
# ===========================================================================

@dataclass
class _ActionContext:
    args: argparse.Namespace
    logger: logging.Logger
    log: _LogAdapter
    progress: _ProgressAdapter
    warning: _WarningAdapter
    input_folder: str
    input_files: list[str]
    file_config: dict[str, Any]


# ===========================================================================
# Section: Core Helpers
# ===========================================================================

def _collect_analysis_files_from_folder(folder: str) -> list[str]:
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {folder}")

    collected: list[str] = []
    for entry in sorted(folder_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _ANALYSIS_SUFFIXES:
            collected.append(str(entry))

    return collected


def _parse_fig_res(value: str) -> list[str]:
    normalized = value.strip().lower().replace(" ", "").replace(",", "x")
    if normalized in _VALID_FIG_RESOLUTIONS:
        return _VALID_FIG_RESOLUTIONS[normalized]
    parts = normalized.split("x")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return parts
    raise ValueError(f"Invalid figure resolution '{value}'. Valid options: {', '.join(_VALID_FIG_RESOLUTIONS)}.")


def _build_options_from_config(config: dict[str, Any], overrides: dict[str, Any]):
    from behavython.pipeline.models import AnalysisOptions

    merged = {**config, **{k: v for k, v in overrides.items() if v is not None}}

    experiment_type = merged.get("experiment_type", "").lower().strip().replace(" ", "_")
    if experiment_type not in _VALID_EXPERIMENT_TYPES:
        raise ValueError(f"Invalid experiment_type '{experiment_type}'. Valid types: {', '.join(sorted(_VALID_EXPERIMENT_TYPES))}.")

    arena_width = int(merged.get("arena_width", 30))
    arena_height = int(merged.get("arena_height", 30))
    fps = int(merged.get("frames_per_second", 30))
    task_duration = int(merged.get("task_duration", 300))
    trim_amount = int(merged.get("trim_amount", 0))
    crop_video = bool(merged.get("crop_video", False))
    generate_video = bool(merged.get("generate_video", False))
    no_plots = bool(merged.get("no_plots", False))
    plot_options = "plotting_disabled" if no_plots else "plotting_enabled"
    algo_type = merged.get("algo_type", "deeplabcut").lower().strip()

    animal = merged.get("animal", "mouse").lower().strip()
    if animal not in _ANIMAL_THRESHOLD:
        raise ValueError(f"Invalid animal '{animal}'. Valid values: {', '.join(_ANIMAL_THRESHOLD)}.")
    threshold = _ANIMAL_THRESHOLD[animal]

    fig_res_raw = merged.get("figure_resolution", "1920x1080")
    max_fig_res = _parse_fig_res(str(fig_res_raw))

    return AnalysisOptions(
        arena_width=arena_width,
        arena_height=arena_height,
        frames_per_second=fps,
        experiment_type=experiment_type,
        max_fig_res=max_fig_res,
        algo_type=algo_type,
        threshold=threshold,
        task_duration=task_duration,
        trim_amount=trim_amount,
        crop_video=crop_video,
        plot_options=plot_options,
        generate_video=generate_video,
    )


def _setup_logging(args: argparse.Namespace) -> tuple[AppLoggingService, logging.Logger]:
    from behavython.core.paths import RUNTIME_ROOT, DATA_ROOT
    from behavython.pipeline.models import RuntimeStorageConfig
    from behavython.services.storage import RuntimeStorage
    from behavython.services.logging import AppLoggingService, register_logging_service

    runtime_storage = RuntimeStorage(
        config=RuntimeStorageConfig(
            runtime_root=RUNTIME_ROOT,
            data_root=DATA_ROOT,
            keep_last_sessions=10,
        )
    )
    app_logging = AppLoggingService(
        runtime_storage,
        is_cli=True,
        console_level=logging.DEBUG if args.verbose else logging.INFO
    )
    register_logging_service(app_logging)
    return app_logging, app_logging.cli_logger


def _load_config(args: argparse.Namespace, logger: logging.Logger) -> dict[str, Any]:
    file_config: dict[str, Any] = {}
    if args.config:
        config_path = os.path.abspath(args.config)
        if not os.path.isfile(config_path):
            logger.error("Config file not found: %s", config_path)
            raise FileNotFoundError(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                file_config = json.load(f)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse config file: %s", exc)
                raise
        logger.info("Loaded config: %s", config_path)
    return file_config


def _resolve_inputs(args: argparse.Namespace, file_config: dict[str, Any], logger: logging.Logger) -> tuple[str, list[str]]:
    input_folder: str = args.input_folder or file_config.get("input_folder", "")
    input_files: list[str] = []

    if input_folder:
        try:
            input_files = _collect_analysis_files_from_folder(input_folder)
        except ValueError as exc:
            logger.error("%s", exc)
            raise

        if not input_files:
            logger.error("No recognizable analysis files found in folder: %s", input_folder)
            logger.error("Expected extensions: %s", ", ".join(_ANALYSIS_SUFFIXES))
            raise ValueError("No recognizable analysis files found in folder.")

        logger.info("Collected %d file(s) from folder: %s", len(input_files), input_folder)

    else:
        input_files = file_config.get("input_files", [])
        if not input_files:
            logger.error("No input source specified. Use --input-folder or set 'input_folder' / 'input_files' in the config.")
            raise ValueError("No input source specified.")

        missing = [p for p in input_files if not os.path.exists(p)]
        if missing:
            logger.error("The following input files do not exist:")
            for p in missing:
                logger.error("  %s", p)
            raise ValueError("One or more required input files do not exist.")

    return input_folder, input_files


# ===========================================================================
# Section: Argument Parsing & Validation
# ===========================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="behavython-cli",
        description=(
            "Run the Behavython analysis pipeline from the command line.\n"
            "All parameters can be set in a JSON config file (--config).\n"
            "CLI flags override matching config file values."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Run full analysis from a config file
  behavython-cli --config run.json

  # Standardize all videos in a folder
  behavython-cli --input-folder ./raw_videos --standardize

  # Run batch cropping from a saved project
  behavython-cli --input-folder ./data --run-cropping --crop-config project.json

  # Run DeepLabCut tracking on a folder
  behavython-cli --input-folder ./videos --run-tracking --dlc-config config.yaml
        """,
    )

    parser.add_argument("--config", "-c", metavar="PATH", help="Path to a JSON config file. See cli_config.example.json for the schema.")
    parser.add_argument(
        "--input-folder",
        "-f",
        metavar="DIR",
        help=(
            "Folder containing all analysis files (CSVs, images, ROI files, JSONs). "
            "Overrides 'input_folder' in config. "
            "Alternatively, set 'input_files' in the config for a hand-picked file list."
        ),
    )
    parser.add_argument("--output", "-o", metavar="DIR", help="Output directory for results. Overrides 'output_folder' in config.")
    parser.add_argument("--arena-config", metavar="PATH", help="Global fallback arena/maze JSON config (required for maze experiments if animals lack individual configs). Overrides 'config_path' in config.")
    parser.add_argument("--no-plots", action="store_true", default=None, help="Disable plot generation. Overrides 'no_plots' in config.")
    parser.add_argument("--experiment-type", metavar="TYPE", help=f"Experiment type. One of: {', '.join(sorted(_VALID_EXPERIMENT_TYPES))}")
    parser.add_argument("--run-tracking", action="store_true", help="Run DeepLabCut tracking (analysis + filtering) on the input videos.")
    parser.add_argument("--dlc-config", metavar="PATH", help="Path to the DeepLabCut config.yaml file (required for --run-tracking).")
    parser.add_argument("--standardize", action="store_true", help="Standardize videos in the input folder to a compatible codec (H.264/CFR).")
    parser.add_argument("--run-cropping", action="store_true", help="Run batch cropping on videos using a previously saved cropping project JSON.")
    parser.add_argument("--crop-config", metavar="PATH", help="Path to the cropping project JSON file (required for --run-cropping).")
    parser.add_argument("--extract-frames", action="store_true", help="Extract one representative frame per video (at 50%% duration by default).")
    parser.add_argument("--frame-number", type=int, metavar="N", default=None, help="Override the frame index used by --extract-frames (default: 50%% of video duration).")
    parser.add_argument("--cleanup", action="store_true", help="Move files not needed for the given experiment type to an 'unwanted_files/' subfolder.")
    parser.add_argument("--likelihood-plots", action="store_true", help="Generate custom likelihood plots for all filtered .h5 tracking files.")
    parser.add_argument("--annotate-video", action="store_true", help="Create DeepLabCut annotated videos.")
    parser.add_argument(
        "--init",
        action="store_true",
        help=(
            "Copy template config files to a directory. Use --experiment-type to get the matching arena config, or omit it to export all templates."
        ),
    )
    parser.add_argument(
        "--transfer-labels",
        action="store_true",
        help=(
            "Transfer DLC labels from one project to another, stripping bodyparts "
            "not present in the target config. Requires --dlc-config and --label-folders."
        ),
    )
    parser.add_argument("--label-folders", nargs="+", metavar="DIR", help="One or more labeled-data directories to transfer (used with --transfer-labels).")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging to stdout.")

    return parser


def _validate_args(args: argparse.Namespace) -> list[str]:
    errors: list[str] = []

    if args.frame_number is not None and not args.extract_frames:
        errors.append("--frame-number has no effect without --extract-frames.")
    if args.crop_config and not args.run_cropping:
        errors.append("--crop-config has no effect without --run-cropping.")
    if args.dlc_config and not any(getattr(args, a, False) for a in _DLC_ACTIONS):
        errors.append("--dlc-config has no effect without a DLC action (--run-tracking, --likelihood-plots, --annotate-video, or --transfer-labels).")
    if args.label_folders and not args.transfer_labels:
        errors.append("--label-folders has no effect without --transfer-labels.")
    if args.no_plots and not any([args.config, args.experiment_type]):
        errors.append("--no-plots has no effect without a behavioral analysis step (--config or --experiment-type).")

    if args.run_cropping and not args.crop_config:
        errors.append("--run-cropping requires --crop-config <PATH>.")
    if args.run_tracking and not args.dlc_config:
        errors.append("--run-tracking requires --dlc-config <PATH>.")
    if args.likelihood_plots and not args.dlc_config:
        errors.append("--likelihood-plots requires --dlc-config <PATH>.")
    if args.annotate_video and not args.dlc_config:
        errors.append("--annotate-video requires --dlc-config <PATH>.")
    if args.transfer_labels and not args.dlc_config:
        errors.append("--transfer-labels requires --dlc-config <PATH> (the TARGET project's config.yaml).")
    if args.transfer_labels and not args.label_folders:
        errors.append("--transfer-labels requires --label-folders <DIR> [DIR ...].")
    if args.cleanup and not args.experiment_type and not args.config:
        errors.append("--cleanup requires --experiment-type <TYPE> (or 'experiment_type' in --config).")

    return errors


# ===========================================================================
# Section: Action Handlers
# ===========================================================================

def _run_init(ctx: _ActionContext) -> bool:
    dest = Path(ctx.args.output).resolve() if ctx.args.output else Path.cwd()
    dest.mkdir(parents=True, exist_ok=True)

    config_src = PACKAGE_ROOT / "config" / "cli_config.example.json"
    if config_src.exists():
        shutil.copy2(config_src, dest / "cli_config.example.json")
        print(f"  [INIT] cli_config.example.json -> {dest}")
    else:
        print(f"  [INIT] WARNING: cli_config.example.json not found at {config_src}", file=sys.stderr)

    arena_map = {
        "open_field": "arena_config_open_field.json",
        "elevated_plus_maze": "arena_config_elevated_plus_maze.json",
    }
    exp_type = ctx.args.experiment_type
    if exp_type:
        targets = {exp_type: arena_map[exp_type]} if exp_type in arena_map else {}
        if not targets:
            print(f"  [INIT] No arena template for experiment type '{exp_type}' (not a maze experiment).")
    else:
        targets = arena_map

    arena_config_root = PACKAGE_ROOT / "assets" / "config"
    for exp, filename in targets.items():
        src = arena_config_root / filename
        if src.exists():
            shutil.copy2(src, dest / filename)
            print(f"  [INIT] {filename} ({exp}) -> {dest}")
        else:
            print(f"  [INIT] WARNING: {filename} not found at {src}", file=sys.stderr)

    # Note: the filename is cli_config.example.json (with two dots), not cli_configexample.json.
    copied_cli = dest / "cli_config.example.json"
    print()
    print("  Files copied to:", dest)
    print()
    if copied_cli.exists():
        print("  cli_config.example.json")
        print("    -> Fill in experiment_type, input_folder, output_folder, fps, etc.")
        print("    -> Then run: behavython-cli --config cli_config.example.json")
        print()
    for exp, filename in targets.items():
        arena_dest = dest / filename
        if arena_dest.exists():
            print(f"  {filename}  [{exp}]")
            print("    -> Fill in the pixel coordinates for your arena/maze.")
            print("    -> Then use it as: behavython-cli --arena-config", arena_dest.name, "--config cli_config.example.json")
            print()

    return True


def _run_transfer_labels(ctx: _ActionContext) -> bool:
    from behavython.scripts.transfer_dlc_labels import transfer_dlc_labels
    output_dir = ctx.args.output or None
    print(f"[TRANSFER] Target config : {ctx.args.dlc_config}")
    print(f"[TRANSFER] Source folders: {len(ctx.args.label_folders)}")
    if output_dir:
        print(f"[TRANSFER] Output dir    : {output_dir}")

    transfer_dlc_labels(
        folders=ctx.args.label_folders,
        target_config_path=ctx.args.dlc_config,
        output_dir=output_dir,
    )
    return True


def _run_standardize(ctx: _ActionContext) -> bool:
    video_extensions = (".mp4", ".avi", ".mov", ".mkv")
    video_files = [f for f in ctx.input_files if f.lower().endswith(video_extensions)]
    if not video_files:
        ctx.logger.error("No video files found for standardization in: %s", ctx.input_folder or "provided files")
        return False

    ctx.logger.info("Videos to process: %d", len(video_files))
    from behavython.services.video_service import run_standardize_videos
    try:
        run_standardize_videos(
            request={"videos": video_files},
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
        ctx.logger.info("Standardization complete.")
    except Exception as exc:
        ctx.logger.error("Standardization failed: %s", exc)
        return False
    return True


def _run_cropping(ctx: _ActionContext) -> bool:
    if not os.path.isfile(ctx.args.crop_config):
        ctx.logger.error("Cropping project JSON not found: %s", ctx.args.crop_config)
        return False

    with open(ctx.args.crop_config, "r", encoding="utf-8") as f:
        try:
            project_data = json.load(f)
        except json.JSONDecodeError as exc:
            ctx.logger.error("Failed to parse crop config: %s", exc)
            return False

    ctx.logger.info("Project: %s", ctx.args.crop_config)
    from behavython.services.video_service import run_batch_crop
    try:
        run_batch_crop(
            request={"project_data": project_data, "project_path": ctx.args.crop_config},
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
        ctx.logger.info("Cropping complete.")
    except Exception as exc:
        ctx.logger.error("Cropping failed: %s", exc)
        return False
    return True


def _run_extract_frames(ctx: _ActionContext) -> bool:
    video_extensions = tuple(ANALYSIS_REQUIRED_SUFFIXES["video"])
    video_files = [f for f in ctx.input_files if f.lower().endswith(video_extensions)]
    if not video_files:
        ctx.logger.error("No video files found for frame extraction in: %s", ctx.input_folder or "provided files")
        return False

    ctx.logger.info("Videos: %d | Frame override: %s", len(video_files), ctx.args.frame_number or "50%% of duration")
    from behavython.pipeline.models import DLCFrameExtractionRequest
    from behavython.pipeline.plugins.dlc import run_extract_frames
    try:
        run_extract_frames(
            request=DLCFrameExtractionRequest(
                video_paths=video_files,
                override_frame_number=ctx.args.frame_number,
            ),
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
        ctx.logger.info("Frame extraction complete.")
    except Exception as exc:
        ctx.logger.error("Frame extraction failed: %s", exc)
        return False
    return True


def _run_cleanup(ctx: _ActionContext) -> bool:
    experiment_type = ctx.args.experiment_type or ctx.file_config.get("experiment_type", "")
    ctx.logger.info("Experiment type: %s", experiment_type)
    ctx.logger.info("Folder: %s", ctx.input_folder or "provided files")

    from behavython.pipeline.models import DLCClearUnusedFilesRequest
    from behavython.pipeline.plugins.dlc import run_clear_unused_files
    try:
        result = run_clear_unused_files(
            request=DLCClearUnusedFilesRequest(
                folder_path=ctx.input_folder,
                task_type=experiment_type,
            ),
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
        ctx.logger.info("Cleanup complete. Moved %d file(s).", len(result.get("moved_files", [])))
        if result.get("missing_files"):
            for mf in result["missing_files"]:
                ctx.logger.warning("Missing required file: %s", mf)
    except Exception as exc:
        ctx.logger.error("Cleanup failed: %s", exc)
        return False
    return True


def _run_tracking(ctx: _ActionContext) -> bool:
    video_extensions = tuple(ANALYSIS_REQUIRED_SUFFIXES["video"])
    video_files = [f for f in ctx.input_files if f.lower().endswith(video_extensions)]
    if not video_files:
        ctx.logger.error("No video files found for tracking in: %s", ctx.input_folder or "provided files")
        return False

    ctx.logger.info("Videos: %d", len(video_files))
    ctx.logger.info("Config: %s", ctx.args.dlc_config)

    from behavython.pipeline.models import DLCVideoAnalysisRequest
    from behavython.pipeline.plugins.dlc import run_dlc_video_analysis
    try:
        run_dlc_video_analysis(
            request=DLCVideoAnalysisRequest(
                config_path=ctx.args.dlc_config,
                video_paths=video_files,
                create_plots=False,
            ),
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
        ctx.logger.info("Tracking task complete.")
    except Exception as exc:
        ctx.logger.error("Tracking failed: %s", exc)
        return False
    return True


def _run_likelihood_plots(ctx: _ActionContext) -> bool:
    ctx.logger.info("Folder: %s", ctx.input_folder or "provided files")
    ctx.logger.info("Config: %s", ctx.args.dlc_config)

    from behavython.pipeline.models import DLCLikelihoodPlotRequest
    from behavython.pipeline.plugins.dlc import run_generate_likelihood_plots
    try:
        result = run_generate_likelihood_plots(
            request=DLCLikelihoodPlotRequest(
                config_path=ctx.args.dlc_config,
                folder_path=ctx.input_folder,
            ),
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
        ctx.logger.info("Generated %d likelihood plot(s).", result.get("plots_generated", 0))
    except Exception as exc:
        ctx.logger.error("Likelihood plot generation failed: %s", exc)
        return False
    return True


def _run_annotate_video(ctx: _ActionContext) -> bool:
    video_extensions = tuple(ANALYSIS_REQUIRED_SUFFIXES["video"])
    video_files = [f for f in ctx.input_files if f.lower().endswith(video_extensions)]
    if not video_files:
        ctx.logger.error("No video files found for annotation in: %s", ctx.input_folder or "provided files")
        return False

    output_folder_annotated = ctx.args.output or ctx.file_config.get("output_folder", ctx.input_folder)
    os.makedirs(output_folder_annotated, exist_ok=True)

    ctx.logger.info("Videos: %d", len(video_files))
    ctx.logger.info("Config: %s", ctx.args.dlc_config)
    ctx.logger.info("Output: %s", output_folder_annotated)

    from behavython.pipeline.models import DLCAnnotatedVideoRequest
    from behavython.pipeline.plugins.dlc import run_create_annotated_video
    try:
        run_create_annotated_video(
            request=DLCAnnotatedVideoRequest(
                config_path=ctx.args.dlc_config,
                video_paths=video_files,
                output_path=output_folder_annotated,
            ),
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
        ctx.logger.info("Annotated video creation complete.")
    except Exception as exc:
        ctx.logger.error("Annotated video creation failed: %s", exc)
        return False
    return True


_ACTION_MAP: dict[str, Callable[[_ActionContext], bool]] = {
    "init": _run_init,
    "transfer_labels": _run_transfer_labels,
    "standardize": _run_standardize,
    "run_cropping": _run_cropping,
    "extract_frames": _run_extract_frames,
    "cleanup": _run_cleanup,
    "run_tracking": _run_tracking,
    "likelihood_plots": _run_likelihood_plots,
    "annotate_video": _run_annotate_video,
}


# ===========================================================================
# Section: Orchestrator
# ===========================================================================

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    arg_errors = _validate_args(args)
    if arg_errors:
        print("[ERROR] Invalid argument combination(s):", file=sys.stderr)
        for err in arg_errors:
            print(f"  \u2022 {err}", file=sys.stderr)
        print("\nRun 'behavython-cli --help' for usage information.", file=sys.stderr)
        return 1

    app_logging, logger = _setup_logging(args)

    if not any(getattr(args, action, False) for action in _TERMINAL_ACTIONS) and not args.config and not args.experiment_type:
        parser.print_help()
        print("\n[ERROR] Provide at least --config or an action flag.", file=sys.stderr)
        return 1

    try:
        file_config = _load_config(args, logger)
    except Exception:
        return 1

    needs_inputs = (
        args.config
        or args.experiment_type
        or any(
            getattr(args, a, False)
            for a in _TERMINAL_ACTIONS
            if a not in _INPUTLESS_ACTIONS
        )
    )

    input_folder = ""
    input_files = []
    if needs_inputs:
        try:
            input_folder, input_files = _resolve_inputs(args, file_config, logger)
        except Exception:
            return 1

    ctx = _ActionContext(
        args=args,
        logger=logger,
        log=_LogAdapter(logger),
        progress=_ProgressAdapter(),
        warning=_WarningAdapter(logger),
        input_folder=input_folder,
        input_files=input_files,
        file_config=file_config,
    )

    for index, action in enumerate(_TERMINAL_ACTIONS):
        if getattr(args, action, False):
            if not _ACTION_MAP[action](ctx):
                return 1
            
            remaining_actions = _TERMINAL_ACTIONS[index + 1:]
            has_remaining = any(getattr(args, a, False) for a in remaining_actions)
            
            if not has_remaining and not args.config and not args.experiment_type:
                return 0

    if not args.config and not args.experiment_type:
        return 0

    output_folder: str = args.output or file_config.get("output_folder", "")
    if not output_folder:
        logger.error("No output folder specified. Use --output or set 'output_folder' in the config.")
        return 1
    os.makedirs(output_folder, exist_ok=True)

    arena_config_path: str | None = args.arena_config or file_config.get("config_path")

    cli_overrides = {
        "experiment_type": args.experiment_type,
        "no_plots": True if args.no_plots else None,
    }
    try:
        options = _build_options_from_config(file_config, cli_overrides)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    from behavython.pipeline.models import AnalysisRequest
    request = AnalysisRequest(
        input_files=input_files,
        output_folder=output_folder,
        options=options,
        config_path=arena_config_path,
    )

    logger.info("Experiment type  : %s", options.experiment_type)
    logger.info("Input files      : %d file(s)", len(input_files))
    logger.info("Output folder    : %s", output_folder)
    logger.info("Plots            : %s", options.plot_options)
    logger.info("Figure resolution: %s x %s px", *options.max_fig_res)

    from behavython.pipeline.workflow import run_analysis_workflow
    from behavython.core.exceptions import AnalysisError

    print()
    try:
        result = run_analysis_workflow(
            request=request,
            progress=ctx.progress,
            log=ctx.log,
            warning=ctx.warning,
        )
    except AnalysisError as exc:
        logger.error("Analysis failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during analysis.")
        return 1

    print()
    logger.info("─" * 50)
    logger.info("Analysis complete.")
    logger.info("  Valid animals   : %d", result.get("valid_animals", 0))
    logger.info("  Invalid animals : %d", result.get("invalid_animals", 0))
    logger.info("  Output path     : %s", result.get("output_path", output_folder))
    if result.get("log_path"):
        logger.warning("Issues were logged to: %s", result["log_path"])

    return 0 if result.get("invalid_animals", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
