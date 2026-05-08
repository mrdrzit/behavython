"""
behavython-cli — Headless command-line entry point for the Behavython analysis pipeline.

This module is completely isolated from PySide6 and Qt.
It constructs an AnalysisRequest from a JSON config file and delegates directly
to run_analysis_workflow(), the same function the GUI uses internally.

Usage:
    behavython-cli --config run.json [--files file1.csv ...] [--output /path/to/out] [--no-plots]

See: cli_config.example.json for the expected config format.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI-compatible signal adapters
# These replace Qt signals (progress.emit, log.emit, warning.emit) with plain
# callables that write to stdout/stderr. The interface is identical so
# run_analysis_workflow() does not need to be modified.
# ---------------------------------------------------------------------------


class _ProgressAdapter:
    """Mimics a Qt Signal with a single .emit(value: int) method."""

    def emit(self, value: int) -> None:
        # Overwrite the current line with a progress bar
        bar_len = 30
        filled = int(bar_len * value / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  Progress [{bar}] {value:3d}%", end="", flush=True)
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
        print(f"\n[WARNING] {title}: {message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Config validation and AnalysisOptions construction
# ---------------------------------------------------------------------------

_VALID_EXPERIMENT_TYPES = {
    "open_field",
    "elevated_plus_maze",
    "social_recognition",
    "social_discrimination",
    "object_discrimination",
}

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

# All suffixes that the analysis pipeline understands.
# Derived from ANALYSIS_REQUIRED_SUFFIXES in defaults.py.
_ANALYSIS_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",  # image
    ".mp4",
    ".avi",
    ".mov",  # video
    ".csv",  # position, skeleton, roi
    ".json",  # maze/arena config
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
  behavython-cli --config run.json
  behavython-cli --config run.json --output /data/results --no-plots
  behavython-cli --config run.json --files a_filtered.csv a.png a_roi.csv
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
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging to stdout.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    # --- Logging setup ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger = logging.getLogger("behavython.cli")

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
    elif not any([args.input_folder, args.output, args.experiment_type]):
        parser.print_help()
        print("\n[ERROR] Provide at least --config or --files + --output + --experiment-type.", file=sys.stderr)
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
