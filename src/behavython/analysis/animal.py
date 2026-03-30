import os
import pandas as pd
import cv2
from src.behavython.config.defaults import BODYPART_MAPPING, CANONICAL_BODYPARTS


class Animal:
    def __init__(self, animal_id, position_csv, skeleton_csv, roi_csv, image_path, video_path=None):
        self.id = animal_id

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
        required = {
            "position_csv": self.position_csv,
            "skeleton_csv": self.skeleton_csv,
            "roi_csv": self.roi_csv,
            "image_path": self.image_path,
        }

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
                self._log(
                    "WARNING",
                    "Multiple video files detected. Using first.",
                    {"files": self.video_path},
                )
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
            self.eligible = False

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
                        self.logs.append({
                            "level": "warning",
                            "message": f"Unmapped bodypart ignored: {bp}",
                            "context": "position_loading"
                        })
                        seen_unknown.add(bp_norm)
                    continue

                canonical = BODYPART_MAPPING[bp_norm]

                if canonical in self.bodyparts:
                    self.logs.append({
                        "level": "warning",
                        "message": f"Duplicate bodypart detected, overwriting: {canonical}",
                        "context": "position_loading"
                    })

                try:
                    bp_df = df[bp]

                    self.bodyparts[canonical] = {
                        "x": bp_df["x"],
                        "y": bp_df["y"],
                        "likelihood": bp_df["likelihood"],
                    }
                except KeyError:
                    self.logs.append({
                        "level": "error",
                        "message": f"Incomplete data for bodypart: {bp}",
                        "context": "position_loading"
                    })

            missing = [
                bp for bp in CANONICAL_BODYPARTS
                if bp not in self.bodyparts
            ]

            if missing:
                self.logs.append({
                    "level": "error",
                    "message": f"Missing required bodyparts: {', '.join(missing)}",
                    "context": "position_loading"
                })
                self.eligible = False

        except Exception as e:
            self.logs.append({
                "level": "error",
                "message": str(e),
                "context": "position_loading"
            })
            self.eligible = False

    def _load_skeleton(self):
        try:
            self.skeleton = {}

            df = pd.read_csv(self.skeleton_csv, header=[0, 1])

            for bone, coord in df.columns:
                if coord != "length":
                    continue

                if bone in self.skeleton:
                    self.logs.append({
                        "level": "warning",
                        "message": f"Duplicate bone detected, overwriting: {bone}",
                        "context": "skeleton_loading"
                    })

                try:
                    bone_df = df[bone]

                    self.skeleton[bone] = {
                        "length": bone_df["length"],
                        "orientation": bone_df["orientation"],
                        "likelihood": bone_df["likelihood"] if "likelihood" in bone_df else None,
                    }

                except KeyError:
                    self.logs.append({
                        "level": "error",
                        "message": f"Incomplete skeleton data: {bone}",
                        "context": "skeleton_loading"
                    })

            # Optional: enforce minimum structure
            if not self.skeleton:
                self.logs.append({
                    "level": "error",
                    "message": "No valid skeleton data found",
                    "context": "skeleton_loading"
                })
                self.eligible = False

        except Exception as e:
            self.logs.append({
                "level": "error",
                "message": "Failed to load skeleton CSV",
                "context": "skeleton_loading",
                "error": str(e),
            })
            self.eligible = False

    def _load_rois(self):
        try:
            df = pd.read_csv(self.roi_csv)

            # Expect columns like: x, y, width, height
            df.columns = [name.lower() for name in df.columns]
            for _, row in df.iterrows():
                roi = {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                }
                self.rois.append(roi)

        except Exception as e:
            self.eligible = False
            self._log(
                "ERROR",
                "Failed to load ROI CSV",
                {"error": str(e)},
            )

    def _load_image(self):
        try:
            self.image = cv2.imread(self.image_path)

            if self.image is None:
                raise ValueError("cv2.imread returned None")

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