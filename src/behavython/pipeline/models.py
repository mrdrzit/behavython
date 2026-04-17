from __future__ import annotations

import logging
import os
import re
import cv2
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from behavython.core.defaults import BODYPART_MAPPING, CANONICAL_BODYPARTS, MAZE_EXPERIMENT_TYPES, TOTAL_SESSION_STORAGE_QUOTA
from behavython.core.exceptions import AnalysisError


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


@dataclass(slots=True)
class DLCFrameExtractionRequest:
    video_paths: list[str]
    override_frame_number: int | None = None


@dataclass(slots=True)
class DLCClearUnusedFilesRequest:
    folder_path: str
    task_type: str


@dataclass(slots=True)
class DLCAnnotatedVideoRequest:
    config_path: str
    video_paths: list[str]
    output_path: str


@dataclass(slots=True)
class RuntimeStorageConfig:
    runtime_root: Path
    data_root: Path
    keep_last_sessions: int = TOTAL_SESSION_STORAGE_QUOTA


@dataclass(slots=True)
class YamlRepairResult:
    success: bool
    original_path: Path
    repaired_path: Path | None
    changed: bool
    message: str
    config: dict[str, Any] | None = None


class MappedFormatter(logging.Formatter):
    def __init__(self, *args, name_map: dict[str, str], **kwargs):
        super().__init__(*args, **kwargs)
        self.name_map = name_map

    def format(self, record):
        original_name = record.name
        record.name = self.name_map.get(record.name, record.name)
        try:
            return super().format(record)
        finally:
            record.name = original_name


class Animal:
    def __init__(self, animal_id, position_csv, image_path, skeleton_csv=None, roi_csv=None, video_path=None, experiment_type="object_recognition"):
        self.id = animal_id
        self.experiment_type = experiment_type

        # File references
        self.position_csv = position_csv
        self.skeleton_csv = skeleton_csv
        self.roi_csv = roi_csv
        self.image_path = image_path
        self.video_path = video_path

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

    def _log(self, level, message, context=None):
        entry = {
            "level": level,
            "message": message,
            "context": context,
        }
        self.logs.append(entry)

    def _validate(self):
        # Base requirements for EVERY analysis
        required = {
            "position_csv": self.position_csv,
            "image_path": self.image_path,
        }

        # Task-specific requirements
        if self.experiment_type not in MAZE_EXPERIMENT_TYPES:
            required["skeleton_csv"] = self.skeleton_csv
            required["roi_csv"] = self.roi_csv

        for key, path in required.items():
            if path is None or not os.path.exists(path):
                self.eligible = False
                self.missing_files.append(key)
                self._log(
                    "ERROR",
                    f"Missing required file: {key}",
                    {"path": path},
                )

        # Video is optional
        if isinstance(self.video_path, list):
            if len(self.video_path) > 1:
                self._log("WARNING", "Multiple video files detected. Using first.", {"files": self.video_path})
                self.video_path = self.video_path[0]
            elif len(self.video_path) == 1:
                self.video_path = self.video_path[0]
            else:
                self.video_path = None

        if not self.eligible:
            self._log(
                "ERROR",
                "Animal marked as NOT ELIGIBLE for analysis due to missing required files",
                {"missing": self.missing_files},
            )

    def _load_all(self):
        self._load_position()
        self._load_skeleton()
        self._load_rois()
        self._load_image()

    def _load_position(self):
        try:
            self.bodyparts = {}
            seen_unknown = set()

            df = pd.read_csv(self.position_csv, header=[0, 1, 2])
            df.columns = df.columns.droplevel(0)

            for bp, coord in df.columns:
                if coord != "x":
                    continue

                bp_norm = bp.strip().lower()

                if bp_norm not in BODYPART_MAPPING:
                    if bp_norm not in seen_unknown:
                        self.logs.append({"level": "warning", "message": f"Unmapped bodypart ignored: {bp}", "context": "position_loading"})
                        seen_unknown.add(bp_norm)
                    continue

                canonical = BODYPART_MAPPING[bp_norm]

                if canonical in self.bodyparts:
                    self.logs.append(
                        {"level": "warning", "message": f"Duplicate bodypart detected, overwriting: {canonical}", "context": "position_loading"}
                    )

                try:
                    bp_df = df[bp]

                    self.bodyparts[canonical] = {
                        "x": bp_df["x"],
                        "y": bp_df["y"],
                        "likelihood": bp_df["likelihood"],
                    }
                except KeyError:
                    self.logs.append({"level": "error", "message": f"Incomplete data for bodypart: {bp}", "context": "position_loading"})

            missing = [bp for bp in CANONICAL_BODYPARTS if bp not in self.bodyparts]

            if missing:
                self.logs.append({"level": "error", "message": f"Missing required bodyparts: {', '.join(missing)}", "context": "position_loading"})
                self.eligible = False

        except Exception as e:
            self.logs.append({"level": "error", "message": str(e), "context": "position_loading"})
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
                    self.logs.append({"level": "warning", "message": f"Duplicate bone detected, overwriting: {bone}", "context": "skeleton_loading"})

                try:
                    bone_df = df[bone]
                    self.skeleton[bone] = {
                        "length": bone_df["length"],
                        "orientation": bone_df["orientation"],
                        "likelihood": bone_df["likelihood"] if "likelihood" in bone_df else None,
                    }
                except KeyError:
                    self.logs.append({"level": "error", "message": f"Incomplete skeleton data: {bone}", "context": "skeleton_loading"})

            if not self.skeleton:
                self.logs.append({"level": "error", "message": "No valid skeleton data found", "context": "skeleton_loading"})
                self.eligible = False

        except Exception as e:
            self.logs.append({"level": "error", "message": "Failed to load skeleton CSV", "context": "skeleton_loading", "error": str(e)})
            self.eligible = False

    def _load_rois(self):
        if not self.roi_csv:
            return

        try:
            df = pd.read_csv(self.roi_csv)
            df.columns = [name.lower().strip() for name in df.columns]

            required_cols = {"x", "y", "bx", "by", "width", "height"}
            missing_cols = required_cols - set(df.columns)
            if missing_cols:
                raise AnalysisError(
                    f"ROI CSV is missing required columns: {', '.join(missing_cols)}. Please check 'Bounding rectangle' and 'Centroid' in ImageJ set measurements when creating the ROIs"
                )

            filename = Path(self.roi_csv).stem
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
            self._log("ERROR", str(e), {"context": "roi_loading"})
        except Exception as e:
            self.eligible = False
            self._log("ERROR", "Failed to load ROI CSV", {"error": str(e)})

    def _load_image(self):
        try:
            self.image = cv2.imread(self.image_path)

            if self.image is None:
                raise AnalysisError("cv2.imread returned None")

        except Exception as e:
            self.eligible = False
            self._log(
                "ERROR",
                "Failed to load image",
                {"error": str(e)},
            )

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
            raise AnalysisError(f"[{self.animal.id}] Base animal data is ineligible for analysis.")

        if self.experiment_type == "open_field" and len(self.arena_corners) != 4:
            raise AnalysisError(f"[{self.animal.id}] Open Field requires exactly 4 arena corners.")

        if self.experiment_type == "elevated_plus_maze" and len(self.maze_points) != 12:
            raise AnalysisError(f"[{self.animal.id}] Elevated Plus Maze requires exactly 12 maze points. Please verify the JSON configuration.")

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
