from __future__ import annotations

import os
import uuid
import shutil
import logging
import subprocess
import numpy as np
import pandas as pd
from typing import Any, List
from natsort import os_sorted
from pathlib import Path
from dataclasses import dataclass, field
from behavython.pipeline.plugins.dlc import load_deeplabcut
from behavython.pipeline.models import DLCAnalyzeFramesRequest
from behavython.core.defaults import ANALYSIS_REQUIRED_SUFFIXES
from behavython.services.logging import capture_external_output
from behavython.core.utils import load_or_repair_dlc_yaml, get_ffmpeg_path, get_ffprobe_path, detect_gpu
from behavython.core.exceptions import AnalysisError, BackupError, BodypartMismatchError, ProjectIntegrityError, ScorerMismatchError

logger = logging.getLogger("behavython.dlc")
console_logger = logging.getLogger("behavython.console")


@dataclass(slots=True)
class ResolvedFrameAssets:
    """Result of discovering and/or extracting frames from the target folder."""

    video_folders: List[Path] = field(default_factory=list)  # Paths to subfolders in session_temp
    newly_extracted: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """True if there are frames available for DLC inference."""
        return len(self.video_folders) > 0


@dataclass(slots=True)
class VideoFolderState:
    """Represents the state of a single folder within labeled-data."""

    folder_name: str
    path: Path
    h5_path: Path | None = None
    csv_path: Path | None = None
    scorer: str | None = None
    bodyparts: list[str] = field(default_factory=list)
    frame_files: list[str] = field(default_factory=list)
    machine_label_files: list[Path] = field(default_factory=list)
    is_valid: bool = False

    def is_compatible_with(self, reference_scorer: str) -> bool:
        """Returns True if this folder has valid labels whose scorer matches the given reference."""
        return self.h5_path is not None and self.is_valid and self.scorer == reference_scorer


@dataclass(slots=True)
class DLCProjectInventory:
    """A snapshot of the entire DLC project's relevant directories."""

    project_root: Path
    config: dict[str, Any]
    labeled_data_path: Path
    video_folders: dict[str, VideoFolderState] = field(default_factory=dict)


def scan_dlc_project(config_path: str | Path) -> DLCProjectInventory:
    """
    Scans the DLC project and returns a structured inventory of its contents.
    This replaces and upgrades the legacy scan_labeled_data_directory.
    """
    config_dict, resolved_config_path, _ = load_or_repair_dlc_yaml(str(config_path))
    project_root = Path(resolved_config_path).parent
    labeled_data_path = project_root / "labeled-data"

    inventory = DLCProjectInventory(project_root=project_root, config=config_dict, labeled_data_path=labeled_data_path)

    if not labeled_data_path.exists():
        raise ProjectIntegrityError(
            f"Labeled-data directory not found: {labeled_data_path}\nThe project folder may have been moved or renamed after training."
        )

    for folder in os.listdir(labeled_data_path):
        folder_path = labeled_data_path / folder
        if not folder_path.is_dir():
            continue

        state = VideoFolderState(folder_name=folder, path=folder_path)

        # Scan files in the video folder
        for file in os.listdir(folder_path):
            file_path = folder_path / file
            if file.startswith("CollectedData_") and file.endswith(".h5"):
                state.h5_path = file_path
                # Extract scorer name from filename as a fallback — will be overridden
                # by the deep H5 read below which is more accurate
                state.scorer = file.replace("CollectedData_", "").replace(".h5", "")
            elif file.startswith("CollectedData_") and file.endswith(".csv"):
                state.csv_path = file_path
            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                state.frame_files.append(file)
            elif file.startswith("machinelabels-iter") and file.endswith(".h5"):
                # DLC writes these after running analyze_frames — track but do not merge
                state.machine_label_files.append(file_path)

        # Deep validation: if we found an H5, let's peek at its bodyparts
        if state.h5_path and state.h5_path.exists():
            try:
                # We only need the header/columns to get bodyparts and verify scorer
                # This is much faster than loading the whole file
                df = pd.read_hdf(state.h5_path, stop=0)
                if isinstance(df.columns, pd.MultiIndex):
                    # DLC columns: scorer, bodyparts, coords
                    state.scorer = df.columns.get_level_values(0).unique()[0]
                    state.bodyparts = list(df.columns.get_level_values(1).unique())
                    state.is_valid = True
            except (OSError, KeyError, ValueError) as e:
                # File is unreadable or malformed — mark as invalid but do not abort the whole scan
                logger.error(f"Could not read H5 metadata from {state.h5_path.name}: {e}")
                state.is_valid = False

        inventory.video_folders[folder] = state

    return inventory


class DLCAssistedLabelSession:
    """
    A transactional session for the Model-Assisted Label Transfer workflow.
    Manages project integrity scanning, backups, temporary files,
    and safe rollback if any step fails.
    """

    def __init__(self, request: DLCAnalyzeFramesRequest):
        self.session_id = uuid.uuid4()
        self.request = request

        # Paths
        self.config_path = Path(request.config_path)
        self.target_folder = Path(request.frames_folder)

        # State
        self.inventory: DLCProjectInventory | None = None
        self.backup_registry: dict[Path, Path] = {}
        self.temp_registry: list[Path] = []
        self.is_committed = False
        self.is_active = False

        # Model metadata — populated during __enter__ from config.yaml
        self.model_metadata: dict[str, Any] = {}

    def __enter__(self) -> DLCAssistedLabelSession:
        self.is_active = True
        logger.info(f"Starting DLC Assisted Label Session {self.session_id}")

        # 1. Initial Scan
        self.inventory = scan_dlc_project(self.config_path)

        # 2. Create session temp directory (inside target frames folder)
        temp_root = self.target_folder / "temp_folder" / f"session_{self.session_id.hex[:8]}"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.session_temp = temp_root

        # 3. Config Lifecycle — extract model metadata and validate compatibility
        self._validate_project_compatibility()

        console_logger.info(f"DLC Session {self.session_id.hex[:8]} initialized | Target: {self.target_folder.name}")

        return self

    def _validate_project_compatibility(self) -> None:
        """
        Extracts model metadata from config.yaml and validates that every
        existing labeled folder is compatible with the current model.
        Raises ScorerMismatchError or BodypartMismatchError on incompatibility.
        """
        config = self.inventory.config

        config_scorer: str = config.get("scorer", "")
        config_bodyparts: list[str] = config.get("bodyparts", [])

        if not config_scorer:
            raise ProjectIntegrityError("The config.yaml does not define a scorer. The project may be corrupted or was not created by DeepLabCut.")

        # Populate model metadata for logging and future-proofing
        self.model_metadata = {
            "scorer": config_scorer,
            "bodyparts": config_bodyparts,
            "network_name": config.get("default_net_type", "unknown"),
            "iteration": config.get("iteration", 0),
            "snapshot_index": config.get("snapshotindex", -1),
        }
        logger.info(
            f"Config validated | scorer={config_scorer} | "
            f"bodyparts={len(config_bodyparts)} | "
            f"network={self.model_metadata['network_name']} | "
            f"iteration={self.model_metadata['iteration']}"
        )

        # Validate every labeled folder that already has data
        for folder_name, folder_state in self.inventory.video_folders.items():
            if not folder_state.is_valid:
                # Folder has no readable labels — skip, it will be treated as new
                continue

            # Scorer check
            if folder_state.scorer != config_scorer:
                raise ScorerMismatchError(
                    f"Scorer mismatch in folder '{folder_name}'.\n"
                    f"  Config scorer : {config_scorer}\n"
                    f"  File scorer   : {folder_state.scorer}\n"
                    "Resolve the scorer conflict before running assisted labeling."
                )

            # Bodypart check — only warn if sets differ, raise if file has extras
            # (missing bodyparts could mean a new bodypart was added to the config,
            #  which is a valid incremental workflow step)
            file_bodyparts = set(folder_state.bodyparts)
            config_bodyparts_set = set(config_bodyparts)

            unexpected = file_bodyparts - config_bodyparts_set
            if unexpected:
                raise BodypartMismatchError(
                    f"Bodypart mismatch in folder '{folder_name}'.\n"
                    f"  Unexpected parts in file (not in config): {sorted(unexpected)}\n"
                    "This may indicate the wrong config.yaml is selected."
                )

    def resolve_frame_assets(self) -> ResolvedFrameAssets:
        """
        Discovers frames in the target folder and optionally extracts them from videos.
        All newly-extracted frames are registered as temp files and cleaned up on rollback.
        """
        frame_ext = self.request.frame_extension.lower()
        if not frame_ext.startswith("."):
            frame_ext = f".{frame_ext}"

        video_files: list[Path] = []
        warnings: list[str] = []

        for item in self.target_folder.iterdir():
            if not item.is_file():
                continue
            suffix = item.suffix.lower()
            if suffix in ANALYSIS_REQUIRED_SUFFIXES["video"]:
                video_files.append(item)

        video_folders = []
        newly_extracted: list[Path] = []

        # 1. Check for existing subfolders (each video is usually its own folder in DLC)
        for item in self.target_folder.iterdir():
            if item.is_dir() and item.name != "temp_folder":
                # Check if it contains frames
                if any(item.glob(f"*{frame_ext}")):
                    # Stage this folder to temp
                    temp_folder = self.session_temp / item.name
                    temp_folder.mkdir(exist_ok=True)
                    for frame in item.glob(f"*{frame_ext}"):
                        shutil.copy2(frame, temp_folder)
                    video_folders.append(temp_folder)

        # 2. Extract new frames from videos (only if in video mode)
        if self.request.mode == "video" and video_files:
            for video in video_files:
                try:
                    # _extract_frames_from_video now creates its own subfolder
                    extracted_folder = self._extract_frames_from_video(video, frame_ext)
                    if extracted_folder:
                        video_folders.append(extracted_folder)
                        newly_extracted.extend(list(extracted_folder.glob(f"*{frame_ext}")))
                except Exception as e:
                    warnings.append(f"Could not extract frames from {video.name}: {e}")
                    logger.warning(f"Frame extraction failed for {video.name}: {e}")

        # 3. Handle images directly in the root folder (only if in image mode)
        if self.request.mode == "image":
            root_images = [item for item in self.target_folder.iterdir() if item.is_file() and item.suffix.lower() == frame_ext]

            if root_images:
                # We treat the root folder itself as a single "video" folder
                # to keep the DLC structure consistent (labeled-data/folder_name/frames)
                root_subfolder = self.session_temp / self.target_folder.name
                root_subfolder.mkdir(exist_ok=True)
                for img in root_images:
                    dest = root_subfolder / img.name
                    shutil.copy2(img, dest)
                    newly_extracted.append(dest)

                if root_subfolder not in video_folders:
                    video_folders.append(root_subfolder)

        if not video_folders:
            warnings.append(f"No valid frame folders, videos, or image assets resolved in {self.target_folder}.")

        console_logger.info(f"Asset resolution | folders_to_analyze={len(video_folders)} | newly_extracted={len(newly_extracted)}")

        return ResolvedFrameAssets(
            video_folders=os_sorted(video_folders),
            newly_extracted=os_sorted(newly_extracted),
            warnings=warnings,
        )

    def _extract_frames_from_video(self, video_path: Path, frame_ext: str) -> Path | None:
        """Extracts evenly-spaced frames from a video into a specific subfolder in temp."""
        video_subfolder = self.session_temp / video_path.stem
        video_subfolder.mkdir(exist_ok=True)

        ffprobe = get_ffprobe_path()
        ffmpeg = get_ffmpeg_path()

        fps_raw = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        num, den = map(float, fps_raw.split("/"))
        fps = num / den

        duration = float(
            subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )

        total_frames = int(duration * fps)
        # number of digits to zero-pad filenames to match DLC convention
        indexlength = len(str(total_frames))
        count = self.request.number_of_frames or 1
        # Evenly space indices, avoiding the very first and last frames
        indices = [int(total_frames * (i + 1) / (count + 1)) for i in range(count)]

        for idx, frame_idx in enumerate(indices):
            timestamp = frame_idx / fps
            out_name = f"img{frame_idx:0{indexlength}d}{frame_ext}"
            # Extract to the video subfolder
            out_path = video_subfolder / out_name

            subprocess.run(
                [ffmpeg, "-ss", str(timestamp), "-i", str(video_path), "-frames:v", "1", "-q:v", "2", "-y", str(out_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.debug(f"Extracted frame {idx + 1}/{count} from {video_path.name} → {out_name}")

        return video_subfolder

    def run_inference(self, assets: ResolvedFrameAssets) -> List[Path]:
        """
        Runs DLC inference on each video folder and transforms results to 'CollectedData' format.
        """
        if not assets.is_ready:
            raise AnalysisError("No assets ready for inference.")

        # GPU detection
        try:
            has_gpu, _ = detect_gpu()
            gpu_to_use = 0 if has_gpu else None
        except Exception:
            gpu_to_use = None

        _, usable_config_path, was_repaired = load_or_repair_dlc_yaml(str(self.config_path))
        deeplabcut = load_deeplabcut()

        final_label_files = []

        for folder in assets.video_folders:
            console_logger.info(f"Analyzing folder: {folder.name}...")

            with capture_external_output("behavython.external"):
                deeplabcut.analyze_time_lapse_frames(
                    str(usable_config_path),
                    str(folder),
                    frametype=self.request.frame_extension,
                    shuffle=1,
                    trainingsetindex=0,
                    gputouse=gpu_to_use,
                    save_as_csv=True,
                )

            # Find the generated H5 (DLC names it based on iterations/date)
            h5_files = list(folder.glob("*.h5"))
            if h5_files:
                # Transform the first one found to CollectedData format
                standardized_h5 = self._transform_to_labeled_data(h5_files[0], folder.name)
                if standardized_h5:
                    final_label_files.append(standardized_h5)

        return final_label_files

    def _transform_to_labeled_data(self, h5_path: Path, folder_name: str) -> Path | None:
        """
        Transforms a machinelabels H5 to the standard DLC 'CollectedData' format.
        - Drops likelihood
        - Renames scorer to match project config
        - Sets MultiIndex [dataset, experiment, image]
        """
        try:
            scorer = self.model_metadata.get("scorer", "unknown")
            bodyparts = self.model_metadata.get("bodyparts", [])
            df = pd.read_hdf(h5_path)

            # 1. Drop likelihood level
            if "likelihood" in df.columns.get_level_values("coords"):
                df = df.drop("likelihood", level="coords", axis=1)

            # 2. Header Construction (Matching DLC create_df_from_prediction logic)
            # We rebuild the columns using from_product to ensure a clean MultiIndex
            # with correct names and levels, sorted as per config bodyparts.
            coords = ["x", "y"]
            cols = [[scorer], bodyparts, coords]
            cols_names = ["scorer", "bodyparts", "coords"]

            # Note: We assume the inference output has all bodyparts defined in config.
            # If not, this from_product approach will fail to align correctly unless
            # we re-index the dataframe.
            new_columns = pd.MultiIndex.from_product(cols, names=cols_names)

            # Align existing data to the new column structure
            # (This handles cases where the model might have different bodypart order)
            df.columns = df.columns.remove_unused_levels()

            # Rebuild a temporary MultiIndex from tuples to ensure we can map values correctly
            # if the scorer name was different in the raw inference file.
            flat_data = []
            target_bps = self.request.target_bodyparts

            for bp in bodyparts:
                for c in coords:
                    if target_bps is not None and bp not in target_bps:
                        # Fill with NaN if bodypart was explicitly excluded by the user
                        flat_data.append(pd.Series(np.nan, index=df.index))
                        continue

                    try:
                        # Try to find data for this bp/coord regardless of old scorer name
                        val = df.xs((bp, c), level=("bodyparts", "coords"), axis=1)
                        flat_data.append(val.iloc[:, 0])
                    except (KeyError, IndexError):
                        # Fill with NaN if bodypart is missing from inference
                        flat_data.append(pd.Series(np.nan, index=df.index))

            df = pd.concat(flat_data, axis=1)
            df.columns = new_columns

            # 3. Fix Index [dataset, experiment, image]
            # Legacy expected: ['labeled-data', folder_name, image_name]
            if isinstance(df.index, pd.MultiIndex):
                image_names = [Path(str(idx[-1])).name for idx in df.index]
            else:
                image_names = [Path(str(idx)).name for idx in df.index]

            new_index = pd.MultiIndex.from_arrays(
                [
                    pd.Series(["labeled-data"] * len(df), dtype=str),
                    pd.Series([str(folder_name)] * len(df), dtype=str),
                    pd.Series([str(n) for n in image_names], dtype=str),
                ],
                names=None,
            )
            df.index = new_index

            # 4. Save standardized files
            dest_h5 = h5_path.parent / f"CollectedData_{scorer}.h5"
            dest_csv = h5_path.parent / f"CollectedData_{scorer}.csv"

            # Use Fixed format for CollectedData (standard for manual labels)
            df.to_hdf(dest_h5, key="df_with_missing", mode="w")
            df.to_csv(dest_csv)

            # Clean up all raw DLC inference files (anything that isn't our final CollectedData)
            for raw_file in h5_path.parent.glob("*"):
                if raw_file.suffix in [".h5", ".csv", ".pickle"]:
                    if not raw_file.name.startswith("CollectedData_"):
                        try:
                            raw_file.unlink()
                        except OSError:
                            pass

            return dest_h5
        except Exception as e:
            logger.error(f"Failed to transform {h5_path.name}: {e}")
            return None

    def _apply_scientific_style(self):
        """Applies publication-ready style with light-weight typography."""
        from behavython.pipeline.plotting import apply_scientific_style

        apply_scientific_style()

    def generate_verification_plots(self, machinelabels_path: Path) -> Path:
        """
        Generates verification plots (annotated frames) for the generated labels.
        This mirrors the legacy 'annotated_frames' functionality.
        Returns the path to the directory containing the plots.
        """
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        # Apply the lightweight typography style
        self._apply_scientific_style()

        plots_dir = self.session_temp / "verification_plots"
        plots_dir.mkdir(exist_ok=True)

        console_logger.info(f"Generating verification plots in {plots_dir.name}...")

        # Load the machinelabels (usually H5)
        df = pd.read_hdf(machinelabels_path)
        scorer = df.columns.get_level_values(0)[0]
        bodyparts = df.columns.get_level_values(1).unique()

        # Find frames in temp folder
        frame_ext = self.request.frame_extension.lower()
        if not frame_ext.startswith("."):
            frame_ext = f".{frame_ext}"

        # Search for frames in all subfolders
        frames = []
        for folder in self.session_temp.iterdir():
            if folder.is_dir() and folder.name != "backups" and folder.name != "verification_plots":
                frames.extend(list(folder.glob(f"*{frame_ext}")))

        frames = sorted(frames)

        # For performance, we limit verification plots to a few samples if there are many
        max_plots = 20
        indices = range(len(frames))
        if len(frames) > max_plots:
            import random

            indices = sorted(random.sample(indices, max_plots))
            console_logger.info(f"Sampling {max_plots} frames for verification plots")

        plt.ioff()  # Disable interactive mode
        for i in indices:
            frame_path = frames[i]
            try:
                img_name = frame_path.name
                session = frame_path.parent.name
                row = df.loc[("labeled-data", session, img_name)]
            except Exception:
                continue

            img = mpimg.imread(frame_path)
            fig, ax = plt.subplots(figsize=(12, 9))
            ax.imshow(img)

            # Use colormap from config or default to Set2 (a nice qualitative pastel-ish map)
            cmap_name = self.inventory.config.get("colormap", "Set2")
            try:
                cmap = plt.get_cmap(cmap_name)
            except Exception:
                cmap = plt.get_cmap("Set2")

            # Sample colors for each bodypart
            bp_colors = [cmap(i / max(1, len(bodyparts) - 1)) for i in range(len(bodyparts))]

            for idx, bp in enumerate(bodyparts):
                x = row[scorer, bp, "x"]
                y = row[scorer, bp, "y"]

                if pd.isna(x) or pd.isna(y):
                    continue

                color = bp_colors[idx]
                ax.plot(x, y, "x", color=color, markersize=4, markeredgewidth=1, label=bp)

            ax.set_title(f"{session} / {frame_path.name}", pad=8)
            ax.axis("off")

            # Add a small, semi-transparent legend inside the plot
            ax.legend(loc="upper right", frameon=True, framealpha=0.4, edgecolor="none")

            plot_path = plots_dir / f"{session}_{frame_path.stem}_verified.png"
            fig.savefig(plot_path, bbox_inches="tight", dpi=120)
            plt.close(fig)

        console_logger.info(f"Verification plots generated: {len(indices)} images")
        return plots_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"DLC Assisted Label Session {self.session_id} failed: {exc_val}")
            self.rollback()
        elif not self.is_committed:
            # If the caller forgot to commit, roll back by default for safety
            logger.warning(f"DLC Assisted Label Session {self.session_id} exited without commit. Rolling back.")
            self.rollback()

        self.cleanup()
        self.is_active = False

    def register_temp(self, path: str | Path):
        """Registers a file or folder for automatic deletion on cleanup."""
        self.temp_registry.append(Path(path))

    def backup(self, path: str | Path):
        """Creates a safe backup of a file before it is modified by the session."""
        original = Path(path)
        if not original.exists() or original in self.backup_registry:
            return

        backup_dir = self.session_temp / "backups"
        backup_dir.mkdir(exist_ok=True)

        backup_path = backup_dir / f"{original.name}.bak"
        try:
            shutil.copy2(original, backup_path)
        except OSError as e:
            raise BackupError(f"Could not back up {original.name} before modifying it.\nReason: {e}\nNo changes will be made to this file.") from e

        self.backup_registry[original] = backup_path
        logger.debug(f"Backed up {original.name} → {backup_path}")

    def commit(self):
        """Finalizes the session, making all changes permanent."""
        if self.is_committed:
            return

        console_logger.info(f"Committing DLC Assisted Label Session {self.session_id.hex[:8]}")

        # Move entire subfolders from temp to target
        subfolders = [f for f in self.session_temp.iterdir() if f.is_dir() and f.name != "backups" and f.name != "verification_plots"]
        moved_count = 0

        for src_folder in subfolders:
            dst_folder = self.target_folder / src_folder.name
            try:
                if dst_folder.exists():
                    # If it exists, we move files inside instead of the whole folder
                    # and back up existing ones
                    for item in src_folder.glob("*"):
                        target_item = dst_folder / item.name
                        if target_item.exists():
                            self.backup(target_item)
                        shutil.move(str(item), str(target_item))
                else:
                    shutil.move(str(src_folder), str(dst_folder))
                moved_count += 1
            except Exception as e:
                logger.error(f"Failed to move folder {src_folder.name} to target: {e}")

        # If verification plots exist, move the whole folder
        plots_dir = self.session_temp / "verification_plots"
        if plots_dir.exists():
            dst_plots = self.target_folder / "verification_plots"
            if dst_plots.exists():
                shutil.rmtree(dst_plots)
            shutil.move(str(plots_dir), str(dst_plots))
            console_logger.info(f"Verification plots saved to {dst_plots.name}")

        self.is_committed = True
        console_logger.info(f"Commit successful | {moved_count} video subfolder(s) transferred to {self.target_folder.name}")

        # 3. Final Synchronization: Run localized convertcsv2h5 on the destination
        # This ensures the H5 files have the perfect structure expected by DLC GUI.
        scorer = self.model_metadata.get("scorer", "unknown")
        for src_folder in subfolders:
            dst_folder = self.target_folder / src_folder.name
            csv_file = dst_folder / f"CollectedData_{scorer}.csv"
            if csv_file.exists():
                try:
                    self._synchronize_csv_to_h5(csv_file, scorer)
                except Exception as e:
                    logger.warning(f"Failed to synchronize {csv_file.name}: {e}")

    def _guarantee_multiindex_rows(self, df: pd.DataFrame):
        """Standard DLC logic to ensure index is a platform-agnostic MultiIndex."""
        if not isinstance(df.index, pd.MultiIndex):
            path = df.index[0]
            try:
                sep = "/" if "/" in path else "\\"
                splits = tuple(df.index.str.split(sep))
                df.index = pd.MultiIndex.from_tuples(splits)
            except (TypeError, AttributeError):
                pass

        try:
            df.index = df.index.set_levels(df.index.levels[1].astype(str), level=1)
        except (AttributeError, IndexError):
            pass

    def _synchronize_csv_to_h5(self, csv_path: Path, scorer: str):
        """Localized version of deeplabcut.convertcsv2h5 that works on any folder."""
        from itertools import islice

        # Determine header depth without loading the whole file
        with open(csv_path) as datafile:
            head = list(islice(datafile, 0, 5))

        if not head:
            return

        if len(head) > 1 and "individuals" in head[1]:
            header = list(range(4))
        else:
            header = list(range(3))

        if head[-1].split(",")[0] == "labeled-data":
            index_col = [0, 1, 2]
        else:
            index_col = 0

        # Read CSV with exact DLC parameters
        data = pd.read_csv(csv_path, index_col=index_col, header=header)

        # Ensure correct scorer level
        try:
            # First try using level name
            data.columns = data.columns.set_levels([scorer], level="scorer")
        except (KeyError, ValueError):
            # Fallback to level index 0
            data.columns = data.columns.set_levels([scorer], level=0)

        # Apply platform-agnostic index formatting
        self._guarantee_multiindex_rows(data)

        # Save to H5 and back to CSV to ensure consistency
        h5_path = str(csv_path).replace(".csv", ".h5")
        data.to_hdf(h5_path, key="df_with_missing", mode="w")
        data.to_csv(csv_path)
        logger.info(f"Synchronized {csv_path.name} to {Path(h5_path).name}")

    def rollback(self):
        """Restores all backed-up files and deletes temporary creations."""
        logger.warning(f"Rolling back DLC Assisted Label Session {self.session_id}")

        for original, backup in self.backup_registry.items():
            try:
                shutil.copy2(backup, original)
                logger.info(f"Restored {original}")
            except Exception as e:
                logger.error(f"Failed to restore {original}: {e}")

    def cleanup(self):
        """Housekeeping: remove temporary files and the session folder."""
        console_logger.info("Cleaning up session temp data...")
        # 1. Delete the specific session folder
        try:
            if hasattr(self, "session_temp") and self.session_temp.exists():
                shutil.rmtree(self.session_temp)
        except Exception as e:
            logger.warning(f"Failed to delete session temp folder: {e}")

        # 2. Delete the parent 'temp_folder' if it's now empty
        try:
            parent_temp = self.target_folder / "temp_folder"
            if parent_temp.exists() and not any(parent_temp.iterdir()):
                parent_temp.rmdir()
                logger.debug("Parent temp_folder removed as it is now empty.")
        except Exception as e:
            logger.debug(f"Could not remove parent temp_folder: {e}")

        # 3. Cleanup individual registered temp items if any outside the session folder
        for path in reversed(self.temp_registry):
            try:
                if hasattr(self, "session_temp") and path == self.session_temp:
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Failed to delete temp item {path}: {e}")
