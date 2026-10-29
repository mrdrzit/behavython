from __future__ import annotations

import logging
import os
import re
import cv2
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from behavython.core.defaults import BODYPART_MAPPING, MAZE_EXPERIMENT_TYPES, ROI_COUNT_BY_EXPERIMENT
from behavython.core.exceptions import AnalysisError

logger = logging.getLogger(__name__)
_ANIMAL_LOG_LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR}


@dataclass(slots=True)
class AnalysisOptions:
    arena_width: int
    arena_height: int
    frames_per_second: int
    experiment_type: str
    max_fig_res: list[str]
    algo_type: str
    threshold: float
    task_duration: int
    trim_amount: int
    crop_video: bool
    plot_options: str
    generate_video: bool = False


@dataclass(slots=True)
class AnalysisRequest:
    input_files: list[str]
    output_folder: str
    options: AnalysisOptions
    config_path: str | None = None


@dataclass(slots=True)
class VideoInputSource:
    folder_path: str = ""
    txt_path: str = ""


@dataclass(slots=True)
class AnalysisInputSource:
    selected_files: list[str]


@dataclass(slots=True)
class OutputFolderSource:
    selected_folder: str = ""


@dataclass(slots=True)
class ResolvedVideoInput:
    paths: list[str] = field(default_factory=list)
    source_kind: str | None = None
    skipped_entries: list[str] = field(default_factory=list)
    duplicate_entries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_paths(self) -> bool:
        return bool(self.paths)


@dataclass(slots=True)
class ResolvedAnalysisInput:
    paths: list[str] = field(default_factory=list)
    skipped_entries: list[str] = field(default_factory=list)
    duplicate_entries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_paths(self) -> bool:
        return bool(self.paths)


@dataclass(slots=True)
class ResolvedOutputFolder:
    path: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def has_path(self) -> bool:
        return bool(self.path)


@dataclass(slots=True)
class DLCVideoAnalysisRequest:
    config_path: str
    video_paths: list[str]
    create_plots: bool = True
    shuffle: int | None = None
    trainingsetindex: int | None = None
    snapshot_index: int | None = None


@dataclass(slots=True)
class DLCFrameExtractionRequest:
    video_paths: list[str]
    override_frame_number: int | None = None


@dataclass(slots=True)
class DLCCreateRoisRequest:
    video_paths: list[str]
    config_path: str


@dataclass(slots=True)
class DLCClearUnusedFilesRequest:
    folder_path: str
    task_type: str


@dataclass(slots=True)
class DLCAnnotatedVideoRequest:
    config_path: str
    video_paths: list[str]
    output_path: str
    snapshot_index: int | None = None


@dataclass(slots=True)
class DLCLikelihoodPlotRequest:
    config_path: str
    folder_path: str
    snapshot_index: int | None = None


@dataclass(slots=True)
class DLCAnalyzeFramesRequest:
    config_path: str
    frames_folder: str
    frame_extension: str = ".jpg"
    number_of_frames: int | None = None
    mode: str = "video"
    target_bodyparts: list[str] | None = None
    shuffle: int | None = None
    trainingsetindex: int | None = None
    snapshot_index: int | None = None


@dataclass(slots=True)
class YamlRepairResult:
    success: bool
    original_path: Path
    repaired_path: Path | None
    changed: bool
    message: str
    config: dict[str, Any] | None = None


class Animal:
    def __init__(
        self,
        animal_id,
        position_csv,
        image_path,
        skeleton_csv=None,
        roi_paths=None,
        video_path=None,
        experiment_type="object_recognition",
        config_path=None,
    ):
        self.id = animal_id
        self.experiment_type = experiment_type

        # File references
        self.position_csv = position_csv
        self.skeleton_csv = skeleton_csv

        # Because list("file.csv") becomes: ['f', 'i', 'l', 'e', '.', 'c', 's', 'v']. Normal
        self.roi_paths = [] if roi_paths is None else [roi_paths] if isinstance(roi_paths, str) else list(roi_paths)
        self.image_path = image_path
        self.video_path = video_path
        self.config_path = config_path

        # Data containers
        self.bodyparts = {}
        self.skeleton = {}
        self.rois = []
        self.image = None

        # State
        self.eligible = True
        self.missing_files = []
        self.logs = []

        # Run validation first
        self._validate()

        # Load only if valid
        if self.eligible:
            self._load_all()

    def log(self, level: str, message: str, context: str) -> None:
        """Records an issue for analysis_log.json and writes it to the session log.

        level is "info", "warning" or "error"; context names the step, for example "position_loading".
        """
        self.logs.append({"level": level, "message": message, "context": context})
        logger.log(_ANIMAL_LOG_LEVELS[level], f"{self.id}: {message}")

    def _validate(self):
        # Base requirements for EVERY analysis
        required = {
            "position_csv": self.position_csv,
            "image_path": self.image_path,
        }

        # Task-specific requirements
        if self.experiment_type not in MAZE_EXPERIMENT_TYPES:
            required["skeleton_csv"] = self.skeleton_csv

        for key, path in required.items():
            if path is None or not os.path.exists(path):
                self.eligible = False
                self.missing_files.append(key)
                self.log("error", f"Missing required file: {key}" + (f" ({path})" if path else ""), "file_validation")

        if self.experiment_type not in MAZE_EXPERIMENT_TYPES:
            if not self.roi_paths:
                self.eligible = False
                self.missing_files.append("roi_paths")
                self.log("error", "Missing required file: roi_paths", "file_validation")
            else:
                expected_rois = ROI_COUNT_BY_EXPERIMENT.get(self.experiment_type)
                if expected_rois is not None and len(self.roi_paths) != expected_rois:
                    self.eligible = False
                    self.missing_files.append("roi_paths")
                    self.log(
                        "error",
                        f"Experiment '{self.experiment_type}' requires exactly {expected_rois} ROI files, found {len(self.roi_paths)}: {self.roi_paths}",
                        "file_validation",
                    )

                for path in self.roi_paths:
                    if path is None or not os.path.exists(path):
                        self.eligible = False
                        self.missing_files.append("roi_paths")
                        self.log("error", f"Missing required ROI file: {path}", "file_validation")

        # Video is optional
        if isinstance(self.video_path, list):
            if len(self.video_path) > 1:
                self.log("warning", f"Multiple video files detected, using the first: {self.video_path}", "file_validation")
                self.video_path = self.video_path[0]
            elif len(self.video_path) == 1:
                self.video_path = self.video_path[0]
            else:
                self.video_path = None

        if not self.eligible:
            self.log("error", f"Not eligible for analysis, missing required files: {', '.join(self.missing_files)}", "file_validation")

    def _load_all(self):
        self._load_position()
        self._load_skeleton()
        self._load_rois()
        self._load_image()

    def _load_position(self):
        try:
            self.bodyparts = {}

            df = pd.read_csv(self.position_csv, header=[0, 1, 2])
            df.columns = df.columns.droplevel(0)

            for bp, coord in df.columns:
                if coord != "x":
                    continue

                bp_norm = bp.strip().lower()

                canonical = BODYPART_MAPPING.get(bp_norm, bp_norm)

                if canonical in self.bodyparts:
                    self.log("warning", f"Duplicate bodypart detected, overwriting: {canonical}", "position_loading")

                try:
                    bp_df = df[bp]

                    self.bodyparts[canonical] = {
                        "x": bp_df["x"],
                        "y": bp_df["y"],
                        "likelihood": bp_df["likelihood"],
                    }
                except KeyError:
                    self.log("error", f"Incomplete data for bodypart: {bp}", "position_loading")

            if self.experiment_type in MAZE_EXPERIMENT_TYPES:
                if not self.bodyparts:
                    self.log("error", "No valid bodyparts found in tracking data", "position_loading")
                    self.eligible = False
            else:
                required = ["nose", "left_ear", "right_ear"]
                missing = [bp for bp in required if bp not in self.bodyparts]

                if missing:
                    self.log("error", f"Missing required bodyparts for {self.experiment_type}: {', '.join(missing)}", "position_loading")
                    self.eligible = False

        except Exception as e:
            self.log("error", str(e), "position_loading")
            self.eligible = False

    def _load_skeleton(self):
        if not self.skeleton_csv:
            return

        try:
            self.skeleton = {}
            df = pd.read_csv(self.skeleton_csv, header=[0, 1])

            for bone, coord in df.columns:
                if coord != "length":
                    continue

                if bone in self.skeleton:
                    self.log("warning", f"Duplicate bone detected, overwriting: {bone}", "skeleton_loading")

                try:
                    bone_df = df[bone]
                    self.skeleton[bone] = {
                        "length": bone_df["length"],
                        "orientation": bone_df["orientation"],
                        "likelihood": bone_df.get("likelihood", None),
                    }
                except KeyError:
                    self.log("error", f"Incomplete skeleton data: {bone}", "skeleton_loading")

            if not self.skeleton:
                self.log("error", "No valid skeleton data found", "skeleton_loading")
                self.eligible = False

        except Exception as e:
            self.log("error", f"Failed to load skeleton CSV: {e}", "skeleton_loading")
            self.eligible = False

    def _load_rois(self):
        if not self.roi_paths:
            return

        for path in self.roi_paths:
            try:
                df = pd.read_csv(path)
                df.columns = [name.lower().strip() for name in df.columns]

                required_cols = {"x", "y", "bx", "by", "width", "height"}
                missing_cols = required_cols - set(df.columns)
                if missing_cols:
                    raise AnalysisError(
                        f"ROI CSV is missing required columns: {', '.join(missing_cols)}. Please check 'Bounding rectangle' and 'Centroid' in ImageJ set measurements when creating the ROIs"
                    )

                filename = Path(path).stem
                match = re.search(r"(roi[lrde]?)$", filename, re.IGNORECASE)
                roi_label = match.group(1) if match else "roi"

                for _, row in df.iterrows():
                    roi = {
                        "name": roi_label,
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "bx": float(row["bx"]),
                        "by": float(row["by"]),
                        "width": float(row["width"]),
                        "height": float(row["height"]),
                    }
                    self.rois.append(roi)

            except AnalysisError as e:
                self.eligible = False
                self.log("error", f"{e} ({path})", "roi_loading")
            except Exception as e:
                self.eligible = False
                self.log("error", f"Failed to load ROI CSV {path}: {e}", "roi_loading")

    def _load_image(self):
        try:
            self.image = cv2.imread(self.image_path)

            if self.image is None:
                raise AnalysisError("cv2.imread returned None")

        except Exception as e:
            self.eligible = False
            self.log("error", f"Failed to load image {self.image_path}: {e}", "image_loading")

    def exp_length(self):
        if not self.bodyparts:
            return 0
        first_bp = next(iter(self.bodyparts.values()))
        return len(first_bp["x"])

    def exp_dimensions(self):
        if self.image is None:
            return (0, 0)
        return self.image.shape[1], self.image.shape[0]


@dataclass
class MazeAnimal:
    """
    A strict geometric validator that wraps the base Animal class.
    Ensures that an animal has the required coordinates and tracking data
    specifically for Open Field or Plus Maze experiments.
    """

    animal: Animal
    experiment_type: str
    arena_corners: list[tuple[float, float]] = field(default_factory=list)
    maze_points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self):
        self._validate_maze_requirements()

    def _validate_maze_requirements(self):
        if not self.animal.eligible:
            raise AnalysisError("Base animal data is ineligible for analysis.")

        if self.experiment_type == "open_field" and len(self.arena_corners) != 4:
            raise AnalysisError("Open Field requires exactly 4 arena corners.")

        if self.experiment_type == "elevated_plus_maze" and len(self.maze_points) != 12:
            raise AnalysisError("Elevated Plus Maze requires exactly 12 maze points. Please verify the JSON configuration.")

    def get_primary_tracking_data(self, preferred_bodypart: str = "center") -> tuple[pd.Series, pd.Series]:
        """
        Retrieves the X and Y series. Falls back to the first available bodypart
        if the preferred one is missing, preventing hardcoded 'focinho' dependencies.
        """
        if preferred_bodypart in self.animal.bodyparts:
            bp_data = self.animal.bodyparts[preferred_bodypart]
        else:
            first_available = next(iter(self.animal.bodyparts))
            bp_data = self.animal.bodyparts[first_available]

        return bp_data["x"], bp_data["y"]
