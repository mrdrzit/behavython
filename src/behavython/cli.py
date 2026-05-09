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
from pathlib import Path
from typing import Any
from behavython.core.defaults import (
    EXPERIMENT_TYPES,
    MAZE_EXPERIMENT_TYPES,
    ANALYSIS_REQUIRED_SUFFIXES,
)


# ---------------------------------------------------------------------------
# CLI-compatible signal adapters
# These replace Qt signals (progress.emit, log.emit, warning.emit) with plain
# callables that write to stdout/stderr. The interface is identical so
# run_analysis_workflow() does not need to be modified.
# ---------------------------------------------------------------------------


class _ProgressAdapter:
    """Mimics a Qt Signal with a single .emit(value: int) method."""

    def emit(self, value: int) -> None:
        # Clear the current line and overwrite with progress bar
        bar_len = 30
        filled = int(bar_len * value / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        # Use \x1b[2K to clear the line and \r to return to start
        sys.stdout.write(f"\r\x1b[2K  Progress [{bar}] {value:3d}%")
        sys.stdout.flush()
        if value >= 100:
            print()  # newline on completion


class _LogAdapter:
    """Mimics a Qt Signal with a .emit(kind: str, message: str) method."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def emit(self, kind: str, message: str) -> None:
        self._logger.info("[%s] %s", kind.upper(), message)


class _WarningAdapter:
    """Mimics a Qt Signal with a .emit(title: str, message: str) method."""

    def emit(self, title: str, message: str) -> None:
        # Avoid duplicate "Warning: Warning" by checking message
        msg = f"{title}: {message}" if title.lower() not in message.lower() else message
        print(f"\r\x1b[2K[WARNING] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Config validation constants — derived from core.defaults to avoid drift
# ---------------------------------------------------------------------------

# All valid experiment types: maze + interaction paradigms
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


# ---------------------------------------------------------------------------
# Folder-based file collection
# ---------------------------------------------------------------------------

# Flat set of file extensions the analysis pipeline understands,
# derived from ANALYSIS_REQUIRED_SUFFIXES so it stays in sync automatically.
_ANALYSIS_SUFFIXES: frozenset[str] = frozenset(
    Path(s).suffix if not s.startswith(".") else s for values in ANALYSIS_REQUIRED_SUFFIXES.values() for s in values
)


def _collect_analysis_files_from_folder(folder: str) -> list[str]:
    """
    Recursively collects all files in 'folder' whose suffixes are known
    to the analysis pipeline. Returns a sorted flat list of absolute paths.

    The caller is responsible for passing this list to group_analysis_files(),
    which handles the animal-ID grouping logic.
    """
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise ValueError(f"Input folder does not exist or is not a directory: {folder}")

    collected: list[str] = []
    for entry in sorted(folder_path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in _ANALYSIS_SUFFIXES:
            collected.append(str(entry))

    return collected


def _parse_fig_res(value: str) -> list[str]:
    """
    Accepts formats like '1920x1080', '1920 x 1080', '1920,1080'.
    Returns ['1920', '1080'].
    """
    normalized = value.strip().lower().replace(" ", "").replace(",", "x")
    if normalized in _VALID_FIG_RESOLUTIONS:
        return _VALID_FIG_RESOLUTIONS[normalized]
    # Try to parse arbitrary WxH
    parts = normalized.split("x")
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return parts
    raise ValueError(f"Invalid figure resolution '{value}'. Valid options: {', '.join(_VALID_FIG_RESOLUTIONS)}.")


def _build_options_from_config(config: dict[str, Any], overrides: dict[str, Any]):
    """
    Constructs an AnalysisOptions dataclass from a merged config dict.
    Raises ValueError with a clear message for any invalid or missing field.
    """
    # Lazy import to avoid pulling in Qt at module load time
    from behavython.pipeline.models import AnalysisOptions

    # Apply CLI flag overrides on top of the config file values
    merged = {**config, **{k: v for k, v in overrides.items() if v is not None}}

    # --- Required fields ---
    experiment_type = merged.get("experiment_type", "").lower().strip().replace(" ", "_")
    if experiment_type not in _VALID_EXPERIMENT_TYPES:
        raise ValueError(f"Invalid experiment_type '{experiment_type}'. Valid types: {', '.join(sorted(_VALID_EXPERIMENT_TYPES))}.")

    # --- Optional with defaults ---
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


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

    parser.add_argument(
        "--config",
        "-c",
        metavar="PATH",
        help="Path to a JSON config file. See cli_config.example.json for the schema.",
    )
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
    parser.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        help="Output directory for results. Overrides 'output_folder' in config.",
    )
    parser.add_argument(
        "--arena-config",
        metavar="PATH",
        help="Global fallback arena/maze JSON config (required for maze experiments if animals lack individual configs). Overrides 'config_path' in config.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=None,
        help="Disable plot generation. Overrides 'no_plots' in config.",
    )
    parser.add_argument(
        "--experiment-type",
        metavar="TYPE",
        help=f"Experiment type. One of: {', '.join(sorted(_VALID_EXPERIMENT_TYPES))}",
    )
    parser.add_argument(
        "--run-tracking",
        action="store_true",
        help="Run DeepLabCut tracking (analysis + filtering) on the input videos.",
    )
    parser.add_argument(
        "--dlc-config",
        metavar="PATH",
        help="Path to the DeepLabCut config.yaml file (required for --run-tracking).",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Standardize videos in the input folder to a compatible codec (H.264/CFR).",
    )
    parser.add_argument(
        "--run-cropping",
        action="store_true",
        help="Run batch cropping on videos using a previously saved cropping project JSON.",
    )
    parser.add_argument(
        "--crop-config",
        metavar="PATH",
        help="Path to the cropping project JSON file (required for --run-cropping).",
    )
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Extract one representative frame per video (at 50%% duration by default).",
    )
    parser.add_argument(
        "--frame-number",
        type=int,
        metavar="N",
        default=None,
        help="Override the frame index used by --extract-frames (default: 50%% of video duration).",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Move files not needed for the given experiment type to an 'unwanted_files/' subfolder.",
    )
    parser.add_argument(
        "--likelihood-plots",
        action="store_true",
        help="Generate custom likelihood plots for all filtered .h5 tracking files in the folder.",
    )
    parser.add_argument(
        "--annotate-video",
        action="store_true",
        help="Create DeepLabCut annotated videos (keypoints overlaid) for all videos in the folder.",
    )
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
    parser.add_argument(
        "--label-folders",
        nargs="+",
        metavar="DIR",
        help="One or more labeled-data directories to transfer (used with --transfer-labels).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to stdout.",
    )

    return parser


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

_DLC_ACTIONS = frozenset(["run_tracking", "likelihood_plots", "annotate_video", "transfer_labels"])


def _validate_args(args: argparse.Namespace) -> list[str]:
    """
    Validates argument combinations upfront, before any actions run.
    Returns a list of human-readable error strings. An empty list means valid.
    """
    errors: list[str] = []

    # --- Orphaned modifiers (flags that require a parent action) ---
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

    # --- Required companions (actions that need a specific flag) ---
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

    # --- Mutually exclusive pairs ---
    # (none currently, but this is where they would go)

    return errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Upfront argument validation (fail fast, fail clearly) ---
    arg_errors = _validate_args(args)
    if arg_errors:
        print("[ERROR] Invalid argument combination(s):", file=sys.stderr)
        for err in arg_errors:
            print(f"  \u2022 {err}", file=sys.stderr)
        print("\nRun 'behavython-cli --help' for usage information.", file=sys.stderr)
        return 1

    # --- ACTION: Init (no logging needed, runs before everything) ---
    if args.init:
        import shutil
        from behavython.core.paths import PACKAGE_ROOT

        dest = Path(args.output).resolve() if args.output else Path.cwd()
        dest.mkdir(parents=True, exist_ok=True)

        # Always copy the CLI config template
        config_src = PACKAGE_ROOT / "config" / "cli_config.example.json"
        if config_src.exists():
            shutil.copy2(config_src, dest / "cli_config.example.json")
            print(f"  [INIT] cli_config.example.json -> {dest}")
        else:
            print(f"  [INIT] WARNING: cli_config.example.json not found at {config_src}", file=sys.stderr)

        # Arena config: export matching type, or all if unspecified
        arena_map = {
            "open_field": "arena_config_open_field.json",
            "elevated_plus_maze": "arena_config_elevated_plus_maze.json",
        }
        exp_type = args.experiment_type
        if exp_type:
            # Export only the matching arena config
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

        # Build a clear, file-specific summary
        copied_cli = dest / "cli_configexample.json"
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

        # --init can be combined with real pipeline actions (e.g. --init --run-tracking).
        # --experiment-type is a modifier, not a standalone action, so it must NOT be counted here
        # (otherwise `--init --experiment-type X` would fall through into input file resolution).
        _other_pipeline_actions = any(
            [
                args.run_tracking,
                args.standardize,
                args.run_cropping,
                args.extract_frames,
                args.cleanup,
                args.likelihood_plots,
                args.annotate_video,
                args.transfer_labels,
                args.config,
            ]
        )
        if not _other_pipeline_actions:
            return 0

    # --- ACTION: Transfer DLC Labels ---
    if args.transfer_labels:
        from behavython.scripts.transfer_dlc_labels import transfer_dlc_labels

        output_dir = args.output or None
        print(f"[TRANSFER] Target config : {args.dlc_config}")
        print(f"[TRANSFER] Source folders: {len(args.label_folders)}")
        if output_dir:
            print(f"[TRANSFER] Output dir    : {output_dir}")

        transfer_dlc_labels(
            folders=args.label_folders,
            target_config_path=args.dlc_config,
            output_dir=output_dir,
        )

        _other_actions = any(
            [
                args.run_tracking,
                args.standardize,
                args.run_cropping,
                args.extract_frames,
                args.cleanup,
                args.likelihood_plots,
                args.annotate_video,
                args.config,
                args.experiment_type,
            ]
        )
        if not _other_actions:
            return 0

    # --- Logging setup: reuse the same AppLoggingService as the GUI ---
    # This gives us: rotating file logs, structured format, filtered external
    # output, and the same noise suppression for TF/Torch/matplotlib.
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
    app_logging = AppLoggingService(runtime_storage)
    register_logging_service(app_logging)

    # The GUI routes console output to a Qt widget. In the CLI we route it
    # to stdout instead so the user sees the same messages in the terminal.
    _cli_console_handler = logging.StreamHandler(sys.stdout)
    _cli_console_handler.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    _cli_console_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    console_logger = logging.getLogger("behavython.console")
    console_logger.addHandler(_cli_console_handler)

    # Also print behavython.cli and behavython.dlc to the terminal
    for _name in ("behavython.cli", "behavython.dlc", "behavython"):
        _lg = logging.getLogger(_name)
        _lg.addHandler(_cli_console_handler)

    logger = logging.getLogger("behavython.cli")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    # --- Load config file ---
    file_config: dict[str, Any] = {}
    if args.config:
        config_path = os.path.abspath(args.config)
        if not os.path.isfile(config_path):
            logger.error("Config file not found: %s", config_path)
            return 1
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                file_config = json.load(f)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse config file: %s", exc)
                return 1
        logger.info("Loaded config: %s", config_path)
    elif not any(
        [
            args.input_folder,
            args.output,
            args.experiment_type,
            args.run_tracking,
            args.standardize,
            args.run_cropping,
            args.extract_frames,
            args.cleanup,
            args.likelihood_plots,
            args.annotate_video,
        ]
    ):
        parser.print_help()
        print("\n[ERROR] Provide at least --config or an action flag.", file=sys.stderr)
        return 1

    # --- Resolve input files ---
    # Priority: CLI --input-folder > config 'input_folder' > config 'input_files' (power-user escape hatch)
    input_folder: str = args.input_folder or file_config.get("input_folder", "")
    input_files: list[str] = []

    if input_folder:
        try:
            input_files = _collect_analysis_files_from_folder(input_folder)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1

        if not input_files:
            logger.error("No recognizable analysis files found in folder: %s", input_folder)
            logger.error("Expected extensions: %s", ", ".join(_ANALYSIS_SUFFIXES))
            return 1

        logger.info("Collected %d file(s) from folder: %s", len(input_files), input_folder)

    else:
        # Fallback: explicit file list from config (not exposed as a CLI flag intentionally)
        input_files = file_config.get("input_files", [])
        if not input_files:
            logger.error("No input source specified. Use --input-folder or set 'input_folder' / 'input_files' in the config.")
            return 1

        # Validate that all explicitly listed files exist
        missing = [p for p in input_files if not os.path.exists(p)]
        if missing:
            logger.error("The following input files do not exist:")
            for p in missing:
                logger.error("  %s", p)
            return 1

    # --- ACTION: Video Standardization ---
    if args.standardize:
        video_extensions = (".mp4", ".avi", ".mov", ".mkv")
        video_files = [f for f in input_files if f.lower().endswith(video_extensions)]

        if not video_files:
            logger.error("No video files found for standardization in: %s", input_folder or "provided files")
            return 1

        logger.info("Videos to process: %d", len(video_files))

        from behavython.services.video_service import run_standardize_videos

        try:
            run_standardize_videos(
                request={"videos": video_files},
                progress=_ProgressAdapter(),
                log=_LogAdapter(logger),
                warning=_WarningAdapter(),
            )
            logger.info("Standardization complete.")
        except Exception as exc:
            logger.error("Standardization failed: %s", exc)
            return 1

        # If this was the only action requested, we exit.
        if not any([args.config, args.experiment_type, args.run_cropping, args.run_tracking]):
            return 0

    # --- ACTION: Batch Cropping ---
    if args.run_cropping:
        if not args.crop_config:
            logger.error("--crop-config is required when using --run-cropping.")
            return 1
        if not os.path.isfile(args.crop_config):
            logger.error("Cropping project JSON not found: %s", args.crop_config)
            return 1

        with open(args.crop_config, "r", encoding="utf-8") as f:
            try:
                project_data = json.load(f)
            except json.JSONDecodeError as exc:
                logger.error("Failed to parse crop config: %s", exc)
                return 1

        logger.info("Project: %s", args.crop_config)

        from behavython.services.video_service import run_batch_crop

        try:
            run_batch_crop(
                request={"project_data": project_data, "project_path": args.crop_config},
                progress=_ProgressAdapter(),
                log=_LogAdapter(logger),
                warning=_WarningAdapter(),
            )
            logger.info("Cropping complete.")
        except Exception as exc:
            logger.error("Cropping failed: %s", exc)
            return 1

        # If this was the last action requested, we exit.
        if not any(
            [args.config, args.experiment_type, args.run_tracking, args.extract_frames, args.cleanup, args.likelihood_plots, args.annotate_video]
        ):
            return 0

    # --- ACTION: Frame Extraction ---
    if args.extract_frames:
        video_extensions = tuple(ANALYSIS_REQUIRED_SUFFIXES["video"])
        video_files = [f for f in input_files if f.lower().endswith(video_extensions)]

        if not video_files:
            logger.error("No video files found for frame extraction in: %s", input_folder or "provided files")
            return 1

        logger.info("Videos: %d | Frame override: %s", len(video_files), args.frame_number or "50%% of duration")

        from behavython.pipeline.models import DLCFrameExtractionRequest
        from behavython.pipeline.plugins.dlc import run_extract_frames

        try:
            run_extract_frames(
                request=DLCFrameExtractionRequest(
                    video_paths=video_files,
                    override_frame_number=args.frame_number,
                ),
                progress=_ProgressAdapter(),
                log=_LogAdapter(logger),
                warning=_WarningAdapter(),
            )
            logger.info("Frame extraction complete.")
        except Exception as exc:
            logger.error("Frame extraction failed: %s", exc)
            return 1

        if not any([args.config, args.experiment_type, args.run_tracking, args.cleanup, args.likelihood_plots, args.annotate_video]):
            return 0

    # --- ACTION: Folder Cleanup ---
    if args.cleanup:
        experiment_type = args.experiment_type or file_config.get("experiment_type", "")
        logger.info("Experiment type: %s", experiment_type)
        logger.info("Folder: %s", input_folder or "provided files")

        from behavython.pipeline.models import DLCClearUnusedFilesRequest
        from behavython.pipeline.plugins.dlc import run_clear_unused_files

        try:
            result = run_clear_unused_files(
                request=DLCClearUnusedFilesRequest(
                    folder_path=input_folder,
                    task_type=experiment_type,
                ),
                progress=_ProgressAdapter(),
                log=_LogAdapter(logger),
                warning=_WarningAdapter(),
            )
            logger.info("Cleanup complete. Moved %d file(s).", len(result.get("moved_files", [])))
            if result.get("missing_files"):
                for mf in result["missing_files"]:
                    logger.warning("Missing required file: %s", mf)
        except Exception as exc:
            logger.error("Cleanup failed: %s", exc)
            return 1

        if not any([args.config, args.experiment_type and not args.cleanup, args.run_tracking, args.likelihood_plots, args.annotate_video]):
            return 0

    # --- ACTION: DeepLabCut Tracking ---
    if args.run_tracking:
        video_extensions = tuple(ANALYSIS_REQUIRED_SUFFIXES["video"])
        video_files = [f for f in input_files if f.lower().endswith(video_extensions)]

        if not video_files:
            logger.error("No video files found for tracking in: %s", input_folder or "provided files")
            return 1

        logger.info("Videos: %d", len(video_files))
        logger.info("Config: %s", args.dlc_config)

        from behavython.pipeline.models import DLCVideoAnalysisRequest
        from behavython.pipeline.plugins.dlc import run_dlc_video_analysis

        tracking_request = DLCVideoAnalysisRequest(
            config_path=args.dlc_config,
            video_paths=video_files,
            create_plots=False,
        )

        try:
            run_dlc_video_analysis(
                request=tracking_request,
                progress=_ProgressAdapter(),
                log=_LogAdapter(logger),
                warning=_WarningAdapter(),
            )
            logger.info("Tracking task complete.")
        except Exception as exc:
            logger.error("Tracking failed: %s", exc)
            return 1

        # If no analysis config is provided, we stop here.
        if not any([args.config, args.experiment_type, args.likelihood_plots, args.annotate_video]):
            logger.info("No analysis configuration provided. Exiting.")
            return 0

    # --- ACTION: Likelihood Plots ---
    if args.likelihood_plots:
        logger.info("Folder: %s", input_folder or "provided files")
        logger.info("Config: %s", args.dlc_config)

        from behavython.pipeline.models import DLCLikelihoodPlotRequest
        from behavython.pipeline.plugins.dlc import run_generate_likelihood_plots

        try:
            result = run_generate_likelihood_plots(
                request=DLCLikelihoodPlotRequest(
                    config_path=args.dlc_config,
                    folder_path=input_folder,
                ),
                progress=_ProgressAdapter(),
                log=_LogAdapter(logger),
                warning=_WarningAdapter(),
            )
            logger.info("Generated %d likelihood plot(s).", result.get("plots_generated", 0))
        except Exception as exc:
            logger.error("Likelihood plot generation failed: %s", exc)
            return 1

        if not any([args.config, args.experiment_type, args.annotate_video]):
            return 0

    # --- ACTION: Create Annotated Video ---
    if args.annotate_video:
        video_extensions = tuple(ANALYSIS_REQUIRED_SUFFIXES["video"])
        video_files = [f for f in input_files if f.lower().endswith(video_extensions)]

        if not video_files:
            logger.error("No video files found for annotation in: %s", input_folder or "provided files")
            return 1

        output_folder_annotated = args.output or file_config.get("output_folder", input_folder)
        os.makedirs(output_folder_annotated, exist_ok=True)

        logger.info("Videos: %d", len(video_files))
        logger.info("Config: %s", args.dlc_config)
        logger.info("Output: %s", output_folder_annotated)

        from behavython.pipeline.models import DLCAnnotatedVideoRequest
        from behavython.pipeline.plugins.dlc import run_create_annotated_video

        try:
            run_create_annotated_video(
                request=DLCAnnotatedVideoRequest(
                    config_path=args.dlc_config,
                    video_paths=video_files,
                    output_path=output_folder_annotated,
                ),
                progress=_ProgressAdapter(),
                log=_LogAdapter(logger),
                warning=_WarningAdapter(),
            )
            logger.info("Annotated video creation complete.")
        except Exception as exc:
            logger.error("Annotated video creation failed: %s", exc)
            return 1

        if not any([args.config, args.experiment_type]):
            return 0

    # --- Resolve output folder ---
    output_folder: str = args.output or file_config.get("output_folder", "")
    if not output_folder:
        logger.error("No output folder specified. Use --output or set 'output_folder' in the config.")
        return 1
    os.makedirs(output_folder, exist_ok=True)

    # --- Resolve arena config path ---
    arena_config_path: str | None = args.arena_config or file_config.get("config_path")

    # --- Build AnalysisOptions ---
    cli_overrides = {
        "experiment_type": args.experiment_type,
        "no_plots": True if args.no_plots else None,
    }
    try:
        options = _build_options_from_config(file_config, cli_overrides)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    # --- Build AnalysisRequest ---
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

    # --- Wire up CLI-compatible callbacks ---
    progress_signal = _ProgressAdapter()
    log_signal = _LogAdapter(logger)
    warning_signal = _WarningAdapter()

    # --- Run ---
    from behavython.pipeline.workflow import run_analysis_workflow
    from behavython.core.exceptions import AnalysisError

    print()
    try:
        result = run_analysis_workflow(
            request=request,
            progress=progress_signal,
            log=log_signal,
            warning=warning_signal,
        )
    except AnalysisError as exc:
        logger.error("Analysis failed: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected error during analysis.")
        return 1

    # --- Report ---
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
