import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class ExtractionConfig:
    extract_frames: bool
    extract_video: bool

    limit_frames: bool
    max_frames: int

    use_txt: bool
    txt_path: str
    threshold: float

    output_directory: str

    use_cuda: bool
    dry_run: bool

    generate_report: bool
    include_ffmpeg_command: bool
    command_max_length: int


# ============================================================================
# INDEX PROVIDERS
# ============================================================================


class IndexProvider:
    """Base class for obtaining frame indices."""

    def get_indices(self, video_path: Path) -> list[int]:
        raise NotImplementedError

    @staticmethod
    def subsample_uniform(
        indices: list[int],
        max_count: int,
    ) -> list[int]:
        """Subsample indices uniformly across time."""
        if not indices or max_count <= 0:
            return []
        if len(indices) <= max_count:
            return list(indices)
        if max_count == 1:
            return [indices[0]]

        n = len(indices)
        sampled_positions = [int(round(i * (n - 1) / (max_count - 1))) for i in range(max_count)]
        seen = set()
        result = []
        for pos in sampled_positions:
            val = indices[pos]
            if val not in seen:
                seen.add(val)
                result.append(val)
        return result


class TxtIndexProvider(IndexProvider):
    """Read frame indices from a TXT file."""

    def __init__(self, txt_path: str):
        self.txt_path = Path(txt_path)

    def get_indices(self, video_path: Path) -> list[int]:
        if not self.txt_path.exists():
            raise FileNotFoundError(f"Indices file not found: {self.txt_path}")

        content = self.txt_path.read_text(encoding="utf-8")

        numbers = re.findall(
            r"\d+",
            content,
        )

        indices = sorted({int(number) for number in numbers if int(number) >= 0})

        if not indices:
            raise ValueError(f"No valid frame indices found in {self.txt_path}")

        return indices


class H5IndexProvider(IndexProvider):
    """Extract low-confidence frame indices from a DeepLabCut H5 file."""

    def __init__(
        self,
        threshold: float,
        log_callback=None,
    ):
        self.threshold = threshold
        self.log_callback = log_callback

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def find_h5(self, video_path: Path) -> Path:
        video_name = video_path.stem

        candidates = sorted(video_path.parent.glob(f"{video_name}*.h5"))

        candidates = [path for path in candidates if not path.name.endswith("_skeleton.h5")]

        def is_valid_match(path: Path) -> bool:
            stem = path.stem
            return stem == video_name or stem.startswith(f"{video_name}DLC") or stem.startswith(f"{video_name}_")

        candidates = [path for path in candidates if is_valid_match(path)]

        if not candidates:
            raise FileNotFoundError(f"No H5 files found for {video_name}")

        filtered = [path for path in candidates if path.name.endswith("_filtered.h5")]

        if filtered:
            return filtered[0]

        return candidates[0]

    def get_indices(self, video_path: Path) -> list[int]:
        h5_path = self.find_h5(video_path)

        self.log(f"Using H5: {h5_path.name}")

        try:
            dataframe = pd.read_hdf(h5_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to read HDF file '{h5_path}': {exc}") from exc

        if not isinstance(
            dataframe.columns,
            pd.MultiIndex,
        ):
            raise ValueError("DLC H5 does not contain MultiIndex columns.")

        if dataframe.columns.nlevels < 2:
            raise ValueError("Unexpected DLC H5 column structure.")

        coordinate_level = dataframe.columns.nlevels - 1

        coordinate_values = dataframe.columns.get_level_values(coordinate_level)

        likelihood_columns = [
            column
            for column, coordinate in zip(
                dataframe.columns,
                coordinate_values,
            )
            if str(coordinate).lower() == "likelihood"
        ]

        if not likelihood_columns:
            raise ValueError("Could not find any 'likelihood' columns in DLC H5.")

        likelihoods = dataframe.loc[
            :,
            likelihood_columns,
        ]

        mask = (likelihoods < self.threshold).any(axis=1)

        indices = []

        for index in dataframe.index[mask]:
            try:
                frame_index = int(index)
            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(f"Could not convert H5 index '{index}' to an integer.") from exc

            indices.append(frame_index)

        indices = sorted(set(indices))

        self.log(f"Likelihood columns: {len(likelihood_columns)}")

        self.log(f"Threshold: {self.threshold:.3f}")

        self.log(f"Low-confidence frames: {len(indices)}")

        return indices


# ============================================================================
# VIDEO INFORMATION
# ============================================================================


class VideoInfo:
    """Retrieve metadata about a video using ffprobe."""

    @staticmethod
    def get_frame_count(
        video_path: Path,
    ) -> int | None:

        command = [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError:
            return None

        if result.returncode != 0:
            return None

        output = result.stdout.strip()

        try:
            return int(output)
        except ValueError:
            return None


# ============================================================================
# FFMPEG
# ============================================================================


class FFmpegExtractor:
    """Build and execute FFmpeg commands."""

    def __init__(
        self,
        use_cuda: bool,
        log_callback=None,
    ):
        self.use_cuda = use_cuda
        self.log_callback = log_callback

        self.process = None
        self.cancel_requested = False

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def cancel(self):
        self.cancel_requested = True

        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass

    # ------------------------------------------------------------------------
    # Frame ranges
    # ------------------------------------------------------------------------

    @staticmethod
    def compress_indices(
        indices: list[int],
    ) -> list[tuple[int, int]]:

        if not indices:
            return []

        ranges = []

        start = indices[0]
        previous = indices[0]

        for index in indices[1:]:
            if index == previous + 1:
                previous = index
                continue

            ranges.append(
                (
                    start,
                    previous,
                )
            )

            start = index
            previous = index

        ranges.append(
            (
                start,
                previous,
            )
        )

        return ranges

    @classmethod
    def build_select_expression(
        cls,
        indices: list[int],
    ) -> str:

        ranges = cls.compress_indices(indices)

        expressions = []

        for start, end in ranges:
            if start == end:
                expressions.append(f"eq(n,{start})")
            else:
                expressions.append(f"between(n,{start},{end})")

        return "+".join(expressions)

    # ------------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------------

    @classmethod
    def create_filter_file(
        cls,
        output_directory: Path,
        indices: list[int],
        filename: str = "select_frames.txt",
    ) -> Path:

        filter_expression = cls.build_select_expression(indices)

        filter_file = output_directory / filename

        filter_contents = f"select='{filter_expression}',setpts=N/FRAME_RATE/TB"

        filter_file.write_text(
            filter_contents,
            encoding="utf-8",
        )

        return filter_file

    # ------------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------------

    def build_base_command(self):
        command = ["ffmpeg"]

        if self.use_cuda:
            command.extend(
                [
                    "-hwaccel",
                    "cuda",
                ]
            )

        return command

    def build_video_command(
        self,
        video_path: Path,
        filter_file: Path,
        output_path: Path,
    ) -> list[str]:

        command = self.build_base_command()

        command.extend(
            [
                "-i",
                str(video_path),
                "-filter_script:v",
                str(filter_file),
            ]
        )

        if self.use_cuda:
            command.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                ]
            )
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                ]
            )

        command.extend(
            [
                "-y",
                str(output_path),
            ]
        )

        return command

    def build_frame_command(
        self,
        video_path: Path,
        filter_file: Path,
        output_pattern: Path,
    ) -> list[str]:

        command = self.build_base_command()

        command.extend(
            [
                "-i",
                str(video_path),
                "-filter_script:v",
                str(filter_file),
                "-fps_mode",
                "vfr",
                "-q:v",
                "2",
                "-y",
                str(output_pattern),
            ]
        )

        return command

    # ------------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------------

    def run_command(
        self,
        command: list[str],
    ) -> bool:

        self.cancel_requested = False

        display_command = " ".join(self.quote_for_display(argument) for argument in command)

        self.log("Executing:")

        self.log(display_command)

        self.log("")

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            assert self.process.stdout is not None

            for line in self.process.stdout:
                if self.cancel_requested:
                    break

                line = line.rstrip()

                if line:
                    self.log(line)

            return_code = self.process.wait()

        except FileNotFoundError:
            self.log("ERROR: FFmpeg was not found.")

            return False

        except Exception as exc:
            self.log(f"FFmpeg execution error: {exc}")

            return False

        finally:
            self.process = None

        if self.cancel_requested:
            self.log("FFmpeg process cancelled.")

            return False

        if return_code != 0:
            self.log(f"FFmpeg failed with exit code {return_code}.")

            return False

        return True

    @staticmethod
    def quote_for_display(
        argument: str,
    ) -> str:

        if " " in argument or "\t" in argument or '"' in argument or "'" in argument:
            return f'"{argument}"'

        return argument

    # ------------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------------

    def extract_video(
        self,
        video_path: Path,
        filter_file: Path,
        output_path: Path,
    ) -> bool:

        command = self.build_video_command(
            video_path,
            filter_file,
            output_path,
        )

        return self.run_command(command)

    def extract_frames(
        self,
        video_path: Path,
        filter_file: Path,
        output_pattern: Path,
    ) -> bool:

        command = self.build_frame_command(
            video_path,
            filter_file,
            output_pattern,
        )

        return self.run_command(command)


# ============================================================================
# OPENCV FRAME EXTRACTOR
# ============================================================================


class OpenCVExtractor:
    """Extract frames directly and rapidly using OpenCV seeking."""

    def __init__(
        self,
        log_callback=None,
    ):
        self.log_callback = log_callback
        self.cancel_requested = False

    def log(self, message: str):
        if self.log_callback:
            self.log_callback(message)

    def cancel(self):
        self.cancel_requested = True

    def extract_frames(
        self,
        video_path: Path,
        indices: list[int],
        output_directory: Path,
        padding: int,
    ) -> bool:
        if not OPENCV_AVAILABLE or cv2 is None:
            self.log("OpenCV is not available.")
            return False

        self.cancel_requested = False
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self.log(f"ERROR: OpenCV could not open video: {video_path}")
            return False

        total_to_extract = len(indices)
        self.log(f"OpenCV: extracting {total_to_extract} frames via direct seeking...")

        extracted = 0
        try:
            for i, index in enumerate(indices, start=1):
                if self.cancel_requested:
                    self.log("OpenCV extraction cancelled.")
                    return False

                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ret, frame = cap.read()
                if not ret or frame is None:
                    self.log(f"WARNING: OpenCV could not read frame {index}")
                    continue

                output_path = output_directory / f"img{index:0{padding}d}.png"
                success = cv2.imwrite(str(output_path), frame)
                if not success:
                    self.log(f"WARNING: Could not write frame to {output_path}")
                    continue

                extracted += 1
                if total_to_extract <= 50 or i % max(1, total_to_extract // 10) == 0 or i == total_to_extract:
                    self.log(f"Extracted {i}/{total_to_extract} frames...")

        finally:
            cap.release()

        self.log(f"OpenCV extraction complete: saved {extracted}/{total_to_extract} frames.")
        return extracted > 0


# ============================================================================
# REPORT GENERATION
# ============================================================================


class ReportGenerator:
    """Generate human-readable analysis reports and FFmpeg command files."""

    @staticmethod
    def format_ranges(
        ranges: list[tuple[int, int]],
    ) -> str:

        if not ranges:
            return "None"

        formatted = []

        for start, end in ranges:
            if start == end:
                formatted.append(str(start))
            else:
                formatted.append(f"{start}-{end}")

        return ", ".join(formatted)

    @staticmethod
    def format_indices(
        indices: list[int],
    ) -> str:

        if not indices:
            return "None"

        return ", ".join(str(index) for index in indices)

    @staticmethod
    def build_report(
        video_path: Path,
        frame_count: int | None,
        threshold: float,
        indices: list[int],
        ranges: list[tuple[int, int]],
        filter_expression: str,
        commands: list[str],
        command_files: list[str],
        extracted_indices: list[int] | None = None,
    ) -> str:

        bad_frames = len(indices)

        if frame_count is not None and frame_count > 0:
            percentage = bad_frames / frame_count * 100

            frame_count_text = str(frame_count)

            percentage_text = f"{percentage:.4f}%"

        else:
            frame_count_text = "Unknown"

            percentage_text = "Unknown"

        lines = []

        lines.append("=" * 70)

        lines.append("DLC LOW-CONFIDENCE FRAME REPORT")

        lines.append("=" * 70)

        lines.append("")

        lines.append(f"Video: {video_path.name}")

        lines.append(f"Path: {video_path}")

        lines.append("")

        lines.append("VIDEO INFORMATION")

        lines.append("-" * 70)

        lines.append(f"Total video frames: {frame_count_text}")

        lines.append(f"Low-confidence frames: {bad_frames}")

        lines.append(f"Bad-frame percentage: {percentage_text}")

        lines.append(f"Likelihood threshold: {threshold:.4f}")

        if extracted_indices is not None and len(extracted_indices) != bad_frames:
            lines.append(f"Frames selected for extraction (uniform temporal sample): {len(extracted_indices)}")

        lines.append("")

        lines.append("LOW-CONFIDENCE FRAME RANGES")

        lines.append("-" * 70)

        lines.append(ReportGenerator.format_ranges(ranges))

        lines.append("")

        lines.append("LOW-CONFIDENCE FRAME INDICES")

        lines.append("-" * 70)

        lines.append(ReportGenerator.format_indices(indices))

        if extracted_indices is not None and len(extracted_indices) != bad_frames:
            lines.append("")

            lines.append("EXTRACTED FRAME INDICES (SUBSAMPLED)")

            lines.append("-" * 70)

            lines.append(ReportGenerator.format_indices(extracted_indices))

        lines.append("")

        lines.append("FFMPEG FILTER EXPRESSION")

        lines.append("-" * 70)

        lines.append(filter_expression)

        if commands:
            lines.append("")

            lines.append("FFMPEG COMMANDS")

            lines.append("-" * 70)

            for command in commands:
                lines.append(command)

                lines.append("")

        if command_files:
            lines.append("")

            lines.append("SEPARATE FFMPEG COMMAND FILES")

            lines.append("-" * 70)

            for command_file in command_files:
                lines.append(command_file)

        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    @staticmethod
    def write_report(
        output_directory: Path,
        video_path: Path,
        content: str,
    ) -> Path:

        report_path = output_directory / f"{video_path.stem}_low_conf_report.txt"

        report_path.write_text(
            content,
            encoding="utf-8",
        )

        return report_path

    @staticmethod
    def write_command_file(
        output_directory: Path,
        video_path: Path,
        command_type: str,
        command: str,
    ) -> Path:

        command_path = output_directory / (f"{video_path.stem}_{command_type}_ffmpeg_command.txt")

        command_path.write_text(
            command,
            encoding="utf-8",
        )

        return command_path


# ============================================================================
# EXTRACTION WORKER
# ============================================================================


class ExtractionWorker(QThread):
    log_msg = Signal(str)

    video_started = Signal(
        str,
        int,
        int,
    )

    video_finished = Signal(
        str,
        bool,
    )

    progress_changed = Signal(int)

    process_finished = Signal(bool)

    process_cancelled = Signal()

    def __init__(
        self,
        video_paths: list[str],
        config: ExtractionConfig,
    ):
        super().__init__()

        self.video_paths = [Path(path) for path in video_paths]

        self.config = config

        self.cancel_requested = False

        self.current_extractor = None

    def log(self, message: str):
        self.log_msg.emit(message)

    def request_cancel(self):

        self.cancel_requested = True

        if self.current_extractor:
            self.current_extractor.cancel()

    @staticmethod
    def cleanup_temporary_frames(output_directory: Path):
        for temp_file in output_directory.glob("_tmp_frame_*.png"):
            try:
                temp_file.unlink()
            except Exception:
                pass

    def create_index_provider(self):

        if self.config.use_txt:
            return TxtIndexProvider(self.config.txt_path)

        return H5IndexProvider(
            threshold=self.config.threshold,
            log_callback=self.log,
        )

    def generate_report(
        self,
        video_path: Path,
        output_directory: Path,
        frame_count: int | None,
        indices: list[int],
        filter_expression: str,
        ranges: list[tuple[int, int]],
        extractor: FFmpegExtractor,
        video_filter_file: Path | None = None,
        frames_filter_file: Path | None = None,
        frame_indices: list[int] | None = None,
    ):

        if not self.config.generate_report:
            return

        commands = []
        command_files = []

        # ------------------------------------------------------------
        # Build commands
        # ------------------------------------------------------------

        if self.config.extract_video and video_filter_file:
            output_video = output_directory / f"{video_path.stem}_low_conf.mp4"

            command = extractor.build_video_command(
                video_path,
                video_filter_file,
                output_video,
            )

            command_text = " ".join(extractor.quote_for_display(argument) for argument in command)

            if self.config.include_ffmpeg_command:
                if len(command_text) <= self.config.command_max_length:
                    commands.append(f"[VIDEO]\n{command_text}")

                else:
                    command_file = ReportGenerator.write_command_file(
                        output_directory,
                        video_path,
                        "video",
                        command_text,
                    )

                    command_files.append(str(command_file))

                    self.log(f"FFmpeg video command is too large for report; saved to {command_file.name}")

        if self.config.extract_frames and frames_filter_file:
            actual_indices = frame_indices if frame_indices else indices
            padding = max(4, len(str(max(actual_indices))))

            output_pattern = output_directory / f"_tmp_frame_%0{padding}d.png"

            command = extractor.build_frame_command(
                video_path,
                frames_filter_file,
                output_pattern,
            )

            command_text = " ".join(extractor.quote_for_display(argument) for argument in command)

            if self.config.include_ffmpeg_command:
                if len(command_text) <= self.config.command_max_length:
                    commands.append(f"[FRAMES]\n{command_text}")

                else:
                    command_file = ReportGenerator.write_command_file(
                        output_directory,
                        video_path,
                        "frames",
                        command_text,
                    )

                    command_files.append(str(command_file))

                    self.log(f"FFmpeg frame command is too large for report; saved to {command_file.name}")

        # ------------------------------------------------------------
        # Report
        # ------------------------------------------------------------

        report_content = ReportGenerator.build_report(
            video_path=video_path,
            frame_count=frame_count,
            threshold=self.config.threshold,
            indices=indices,
            ranges=ranges,
            filter_expression=filter_expression,
            commands=commands,
            command_files=command_files,
            extracted_indices=frame_indices if (self.config.extract_frames and self.config.limit_frames) else None,
        )

        report_path = ReportGenerator.write_report(
            output_directory,
            video_path,
            report_content,
        )

        self.log(f"Report created: {report_path}")

    def process_video(
        self,
        video_path: Path,
    ) -> bool:

        self.log("")
        self.log("=" * 70)

        self.log(f"PROCESSING: {video_path.name}")

        self.log("=" * 70)

        if not video_path.exists():
            self.log(f"ERROR: Video does not exist: {video_path}")

            return False

        # ------------------------------------------------------------
        # Output directory
        # ------------------------------------------------------------

        if self.config.output_directory:
            output_directory = Path(self.config.output_directory) / video_path.stem

        else:
            output_directory = video_path.parent / video_path.stem

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.log(f"Output directory: {output_directory}")

        # ------------------------------------------------------------
        # Video frame count
        # ------------------------------------------------------------

        frame_count = VideoInfo.get_frame_count(video_path)

        if frame_count is not None:
            self.log(f"Video frame count: {frame_count}")

        else:
            self.log("WARNING: Could not determine video frame count with ffprobe.")

        # ------------------------------------------------------------
        # Indices
        # ------------------------------------------------------------

        provider = self.create_index_provider()

        try:
            indices = provider.get_indices(video_path)

        except Exception as exc:
            self.log(f"ERROR: {exc}")

            return False

        if not indices:
            self.log("No low-confidence frames found.")

            # Still create a report.
            if self.config.generate_report:
                extractor = FFmpegExtractor(
                    use_cuda=self.config.use_cuda,
                    log_callback=self.log,
                )

                self.generate_report(
                    video_path=video_path,
                    output_directory=output_directory,
                    frame_count=frame_count,
                    indices=[],
                    filter_expression="",
                    ranges=[],
                    extractor=extractor,
                )

            return True

        # ------------------------------------------------------------
        # Validate indices against video
        # ------------------------------------------------------------

        if frame_count is not None:
            invalid_indices = [index for index in indices if index >= frame_count]

            if invalid_indices:
                self.log(f"WARNING: {len(invalid_indices)} indices are outside the video frame range.")

                self.log(f"First invalid index: {invalid_indices[0]}")

                indices = [index for index in indices if index < frame_count]

        if not indices:
            self.log("No valid indices remain after video-frame validation.")

            return True

        # ------------------------------------------------------------
        # Ranges
        # ------------------------------------------------------------

        ranges = FFmpegExtractor.compress_indices(indices)

        filter_expression = FFmpegExtractor.build_select_expression(indices)

        self.log(f"Low-confidence frames: {len(indices)}")

        self.log(f"Frame range: {indices[0]} - {indices[-1]}")

        self.log(f"Compressed into {len(ranges)} ranges.")

        # ------------------------------------------------------------
        # Report-only mode (neither frames nor video extraction requested)
        # ------------------------------------------------------------

        if not self.config.extract_frames and not self.config.extract_video:
            if self.config.generate_report:
                extractor = FFmpegExtractor(
                    use_cuda=self.config.use_cuda,
                    log_callback=self.log,
                )
                try:
                    self.generate_report(
                        video_path=video_path,
                        output_directory=output_directory,
                        frame_count=frame_count,
                        indices=indices,
                        filter_expression=filter_expression,
                        ranges=ranges,
                        extractor=extractor,
                        video_filter_file=None,
                        frames_filter_file=None,
                        frame_indices=None,
                    )
                except Exception as exc:
                    self.log(f"WARNING: Could not generate report: {exc}")

            self.log("Report-only mode complete. FFmpeg extraction skipped.")
            return True

        # ------------------------------------------------------------
        # Frame subsampling if limit is enabled
        # ------------------------------------------------------------

        frame_indices = indices
        if self.config.extract_frames and self.config.limit_frames and len(indices) > self.config.max_frames:
            frame_indices = IndexProvider.subsample_uniform(indices, self.config.max_frames)
            self.log(f"Frame limit active: extracting {len(frame_indices)} frames (subsampled uniformly from {len(indices)} detected frames).")

        # ------------------------------------------------------------
        # Filter files
        # ------------------------------------------------------------

        video_filter_file = None
        if self.config.extract_video:
            try:
                video_filter_file = FFmpegExtractor.create_filter_file(
                    output_directory,
                    indices,
                    filename="select_video.txt",
                )
            except Exception as exc:
                self.log(f"ERROR creating FFmpeg video filter: {exc}")
                return False

        frames_filter_file = None
        if self.config.extract_frames and not OPENCV_AVAILABLE:
            try:
                frames_filter_file = FFmpegExtractor.create_filter_file(
                    output_directory,
                    frame_indices,
                    filename="select_frames.txt",
                )
            except Exception as exc:
                self.log(f"ERROR creating FFmpeg frames filter: {exc}")
                return False

        # ------------------------------------------------------------
        # FFmpeg extractor
        # ------------------------------------------------------------

        extractor = FFmpegExtractor(
            use_cuda=self.config.use_cuda,
            log_callback=self.log,
        )

        self.current_extractor = extractor

        # ------------------------------------------------------------
        # Generate report BEFORE execution
        #
        # This means the report is also useful in dry-run mode.
        # ------------------------------------------------------------

        try:
            self.generate_report(
                video_path=video_path,
                output_directory=output_directory,
                frame_count=frame_count,
                indices=indices,
                filter_expression=filter_expression,
                ranges=ranges,
                extractor=extractor,
                video_filter_file=video_filter_file,
                frames_filter_file=frames_filter_file,
                frame_indices=frame_indices,
            )

        except Exception as exc:
            self.log(f"WARNING: Could not generate report: {exc}")

        # ------------------------------------------------------------
        # Dry run
        # ------------------------------------------------------------

        if self.config.dry_run:
            self.log("")
            self.log("DRY RUN — Media extraction will not be executed.")

            if self.config.extract_frames:
                self.log(f"Would extract {len(frame_indices)} frames.")
            if self.config.extract_video:
                self.log(f"Would extract {len(indices)} video frames.")

            for filter_path in (video_filter_file, frames_filter_file):
                if filter_path and filter_path.exists():
                    try:
                        filter_path.unlink()
                    except Exception:
                        pass

            self.current_extractor = None

            return True

        # ------------------------------------------------------------
        # Extraction
        # ------------------------------------------------------------

        success = True

        try:
            # --------------------------------------------------------
            # Video
            # --------------------------------------------------------

            if self.config.extract_video and video_filter_file:
                output_video = output_directory / f"{video_path.stem}_low_conf.mp4"

                self.log("")
                self.log("EXTRACTING VIDEO")

                video_success = extractor.extract_video(
                    video_path=video_path,
                    filter_file=video_filter_file,
                    output_path=output_video,
                )

                if not video_success:
                    success = False

                elif output_video.exists():
                    self.log(f"Created: {output_video}")

            # --------------------------------------------------------
            # Frames
            # --------------------------------------------------------

            if self.config.extract_frames and not self.cancel_requested:
                self.log("")
                self.log("EXTRACTING FRAMES")

                padding = max(
                    4,
                    len(str(max(frame_indices))),
                )

                if OPENCV_AVAILABLE:
                    opencv_extractor = OpenCVExtractor(log_callback=self.log)
                    self.current_extractor = opencv_extractor

                    frames_success = opencv_extractor.extract_frames(
                        video_path=video_path,
                        indices=frame_indices,
                        output_directory=output_directory,
                        padding=padding,
                    )

                    if not frames_success:
                        success = False

                else:
                    self.log("OpenCV not available; falling back to FFmpeg frame extraction...")

                    if frames_filter_file:
                        self.cleanup_temporary_frames(output_directory)

                        temporary_pattern = output_directory / f"_tmp_frame_%0{padding}d.png"

                        frames_success = extractor.extract_frames(
                            video_path=video_path,
                            filter_file=frames_filter_file,
                            output_pattern=temporary_pattern,
                        )

                        if not frames_success:
                            success = False

                        else:
                            rename_success = self.rename_frames(
                                output_directory,
                                frame_indices,
                                padding,
                            )

                            if not rename_success:
                                success = False

        finally:
            for filter_path in (video_filter_file, frames_filter_file):
                if filter_path and filter_path.exists():
                    try:
                        filter_path.unlink()
                    except Exception:
                        pass

            if self.cancel_requested:
                self.cleanup_temporary_frames(output_directory)
            self.current_extractor = None

        return success

    def rename_frames(
        self,
        output_directory: Path,
        indices: list[int],
        padding: int,
    ) -> bool:

        extracted_frames = sorted(output_directory.glob("_tmp_frame_*.png"))

        self.log(f"FFmpeg produced {len(extracted_frames)} frames.")

        if len(extracted_frames) != len(indices):
            self.log("ERROR: Number of extracted frames does not match number of requested indices.")

            self.log(f"Expected: {len(indices)}")

            self.log(f"Received: {len(extracted_frames)}")

            self.cleanup_temporary_frames(output_directory)

            return False

        self.log("Renaming frames...")

        try:
            for index, temporary_file in zip(
                indices,
                extracted_frames,
            ):
                destination = output_directory / f"img{index:0{padding}d}.png"

                if destination.exists():
                    destination.unlink()

                temporary_file.rename(destination)

        except Exception as exc:
            self.log(f"ERROR renaming frames: {exc}")

            self.cleanup_temporary_frames(output_directory)

            return False

        self.log(f"Created {len(indices)} PNG frames.")

        return True

    def run(self):

        total = len(self.video_paths)

        successful = 0

        self.log(f"Starting batch: {total} video(s)")

        for number, video_path in enumerate(
            self.video_paths,
            start=1,
        ):
            if self.cancel_requested:
                break

            self.video_started.emit(
                video_path.name,
                number,
                total,
            )

            success = self.process_video(video_path)

            if success:
                successful += 1

            self.video_finished.emit(
                video_path.name,
                success,
            )

            progress = int(number / total * 100)

            self.progress_changed.emit(progress)

        if self.cancel_requested:
            self.log("")
            self.log("BATCH CANCELLED.")

            self.process_cancelled.emit()

            return

        success = successful == total

        self.log("")
        self.log("=" * 70)

        self.log(f"BATCH COMPLETE: {successful}/{total} successful")

        self.log("=" * 70)

        self.process_finished.emit(success)


# ============================================================================
# MAIN WINDOW
# ============================================================================


class MainWindow(QWidget):
    def __init__(self):

        super().__init__()

        self.worker = None

        self.setWindowTitle("DLC Low-Confidence Frame Extractor")

        self.resize(
            1100,
            750,
        )

        self.init_ui()

    # ------------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------------

    def init_ui(self):

        main_layout = QHBoxLayout(self)

        # ====================================================================
        # LEFT PANEL
        # ====================================================================

        left_layout = QVBoxLayout()

        left_layout.addWidget(QLabel("<b>Video Queue</b>"))

        self.list_widget = QListWidget()

        self.list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)

        left_layout.addWidget(self.list_widget)

        queue_buttons = QHBoxLayout()

        self.btn_add = QPushButton("Add Video(s)")

        self.btn_remove = QPushButton("Remove Selected")

        self.btn_clear = QPushButton("Clear All")

        self.btn_add.clicked.connect(self.add_videos)

        self.btn_remove.clicked.connect(self.remove_videos)

        self.btn_clear.clicked.connect(self.clear_videos)

        queue_buttons.addWidget(self.btn_add)

        queue_buttons.addWidget(self.btn_remove)

        queue_buttons.addWidget(self.btn_clear)

        left_layout.addLayout(queue_buttons)

        # ====================================================================
        # RIGHT PANEL
        # ====================================================================

        right_layout = QVBoxLayout()

        # --------------------------------------------------------------------
        # Extraction
        # --------------------------------------------------------------------

        extraction_group = QGroupBox("Extraction")

        extraction_layout = QVBoxLayout()

        self.chk_extract_frames = QCheckBox("Extract Frames (PNG)")

        self.chk_extract_frames.setChecked(True)

        limit_layout = QHBoxLayout()

        limit_layout.setContentsMargins(20, 0, 0, 0)

        self.chk_limit_frames = QCheckBox("Limit extracted frames (uniform temporal spacing)")

        self.chk_limit_frames.setChecked(False)

        self.spin_max_frames = QSpinBox()

        self.spin_max_frames.setRange(1, 100_000)

        self.spin_max_frames.setValue(100)

        self.spin_max_frames.setSuffix(" frames")

        limit_layout.addWidget(self.chk_limit_frames)

        limit_layout.addWidget(self.spin_max_frames)

        limit_layout.addStretch()

        self.chk_extract_video = QCheckBox("Extract Video (MP4)")

        extraction_layout.addWidget(self.chk_extract_frames)

        extraction_layout.addLayout(limit_layout)

        extraction_layout.addWidget(self.chk_extract_video)

        extraction_group.setLayout(extraction_layout)

        self.chk_extract_frames.toggled.connect(self.update_ui_state)

        self.chk_limit_frames.toggled.connect(self.update_ui_state)

        # --------------------------------------------------------------------
        # Source
        # --------------------------------------------------------------------

        source_group = QGroupBox("Indices Source")

        source_layout = QVBoxLayout()

        self.radio_h5 = QRadioButton("Auto-detect H5 (DeepLabCut)")

        self.radio_h5.setChecked(True)

        threshold_layout = QHBoxLayout()

        threshold_layout.addWidget(QLabel("Likelihood threshold:"))

        self.spin_threshold = QDoubleSpinBox()

        self.spin_threshold.setRange(
            0.0,
            1.0,
        )

        self.spin_threshold.setSingleStep(0.05)

        self.spin_threshold.setDecimals(2)

        self.spin_threshold.setValue(0.80)

        threshold_layout.addWidget(self.spin_threshold)

        threshold_layout.addStretch()

        self.radio_txt = QRadioButton("Use custom TXT indices")

        txt_layout = QHBoxLayout()

        self.txt_path_edit = QLineEdit()

        self.txt_path_edit.setPlaceholderText("Select .txt file...")

        self.txt_path_edit.setReadOnly(True)

        self.btn_browse_txt = QPushButton("Browse")

        self.btn_browse_txt.clicked.connect(self.browse_txt)

        txt_layout.addWidget(self.txt_path_edit)

        txt_layout.addWidget(self.btn_browse_txt)

        source_layout.addWidget(self.radio_h5)

        source_layout.addLayout(threshold_layout)

        source_layout.addWidget(self.radio_txt)

        source_layout.addLayout(txt_layout)

        source_group.setLayout(source_layout)

        self.radio_h5.toggled.connect(self.update_ui_state)

        # --------------------------------------------------------------------
        # Output
        # --------------------------------------------------------------------

        output_group = QGroupBox("Output")

        output_layout = QVBoxLayout()

        output_path_layout = QHBoxLayout()

        output_path_layout.addWidget(QLabel("Directory:"))

        self.output_path_edit = QLineEdit()

        self.output_path_edit.setPlaceholderText("Empty = <video_name> folder next to each video")

        self.btn_browse_output = QPushButton("Browse")

        self.btn_browse_output.clicked.connect(self.browse_output)

        output_path_layout.addWidget(self.output_path_edit)

        output_path_layout.addWidget(self.btn_browse_output)

        output_layout.addLayout(output_path_layout)

        self.chk_cuda = QCheckBox("Use CUDA hardware decoding")

        self.chk_cuda.setChecked(True)

        self.chk_dry_run = QCheckBox("Dry run (do not execute FFmpeg)")

        output_layout.addWidget(self.chk_cuda)

        output_layout.addWidget(self.chk_dry_run)

        output_group.setLayout(output_layout)

        # --------------------------------------------------------------------
        # Reports
        # --------------------------------------------------------------------

        report_group = QGroupBox("Reports")

        report_layout = QVBoxLayout()

        self.chk_report = QCheckBox("Generate analysis report")

        self.chk_report.setChecked(True)

        self.chk_commands = QCheckBox("Include standalone FFmpeg command")

        self.chk_commands.setChecked(True)

        command_size_layout = QHBoxLayout()

        command_size_layout.addWidget(QLabel("Maximum command length:"))

        self.spin_command_size = QSpinBox()

        self.spin_command_size.setRange(
            1000,
            1_000_000,
        )

        self.spin_command_size.setSingleStep(1000)

        self.spin_command_size.setValue(8000)

        self.spin_command_size.setSuffix(" characters")

        command_size_layout.addWidget(self.spin_command_size)

        command_size_layout.addStretch()

        report_layout.addWidget(self.chk_report)

        report_layout.addWidget(self.chk_commands)

        report_layout.addLayout(command_size_layout)

        report_group.setLayout(report_layout)

        self.chk_report.toggled.connect(self.update_report_ui)

        # --------------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------------

        self.current_video_label = QLabel("Idle")

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            100,
        )

        self.progress_bar.setValue(0)

        # --------------------------------------------------------------------
        # Run
        # --------------------------------------------------------------------

        run_layout = QHBoxLayout()

        self.btn_run = QPushButton("Run Extraction")

        self.btn_run.setStyleSheet(
            """
            QPushButton {
                font-weight: bold;
                padding: 10px;
            }
            """
        )

        self.btn_cancel = QPushButton("Cancel")

        self.btn_cancel.setEnabled(False)

        self.btn_run.clicked.connect(self.run_extraction)

        self.btn_cancel.clicked.connect(self.cancel_extraction)

        run_layout.addWidget(self.btn_run)

        run_layout.addWidget(self.btn_cancel)

        # --------------------------------------------------------------------
        # Log
        # --------------------------------------------------------------------

        self.log_text = QTextEdit()

        self.log_text.setReadOnly(True)

        self.log_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
            }
            """
        )

        # --------------------------------------------------------------------
        # Assemble
        # --------------------------------------------------------------------

        right_layout.addWidget(extraction_group)

        right_layout.addWidget(source_group)

        right_layout.addWidget(output_group)

        right_layout.addWidget(report_group)

        right_layout.addWidget(self.current_video_label)

        right_layout.addWidget(self.progress_bar)

        right_layout.addLayout(run_layout)

        right_layout.addWidget(QLabel("<b>Log</b>"))

        right_layout.addWidget(self.log_text)

        main_layout.addLayout(
            left_layout,
            1,
        )

        main_layout.addLayout(
            right_layout,
            2,
        )

        self.update_ui_state()
        self.update_report_ui(self.chk_report.isChecked())

    # ------------------------------------------------------------------------
    # UI STATE
    # ------------------------------------------------------------------------

    def update_ui_state(self):

        use_txt = self.radio_txt.isChecked()

        self.txt_path_edit.setEnabled(use_txt)

        self.btn_browse_txt.setEnabled(use_txt)

        self.spin_threshold.setEnabled(not use_txt)

        frames_enabled = self.chk_extract_frames.isChecked()

        self.chk_limit_frames.setEnabled(frames_enabled)

        self.spin_max_frames.setEnabled(frames_enabled and self.chk_limit_frames.isChecked())

    def update_report_ui(
        self,
        enabled: bool,
    ):

        self.chk_commands.setEnabled(enabled)

        self.spin_command_size.setEnabled(enabled and self.chk_commands.isChecked())

    # ------------------------------------------------------------------------
    # Queue
    # ------------------------------------------------------------------------

    def add_videos(self):

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video(s)",
            "",
            ("Videos (*.mp4 *.avi *.mov *.mkv *.wmv)"),
        )

        for file_path in files:
            existing = self.list_widget.findItems(
                file_path,
                Qt.MatchExactly,
            )

            if not existing:
                self.list_widget.addItem(file_path)

    def remove_videos(self):

        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))

    def clear_videos(self):

        self.list_widget.clear()

    # ------------------------------------------------------------------------
    # Browsers
    # ------------------------------------------------------------------------

    def browse_txt(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Indices TXT",
            "",
            "Text Files (*.txt)",
        )

        if file_path:
            self.txt_path_edit.setText(file_path)

    def browse_output(self):

        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
        )

        if directory:
            self.output_path_edit.setText(directory)

    # ------------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------------

    def append_log(
        self,
        text: str,
    ):

        self.log_text.append(text)

        scrollbar = self.log_text.verticalScrollBar()

        scrollbar.setValue(scrollbar.maximum())

    # ------------------------------------------------------------------------
    # Running state
    # ------------------------------------------------------------------------

    def set_running_state(
        self,
        running: bool,
    ):

        self.btn_run.setEnabled(not running)

        self.btn_cancel.setEnabled(running)

        self.btn_add.setEnabled(not running)

        self.btn_remove.setEnabled(not running)

        self.btn_clear.setEnabled(not running)

        self.chk_extract_frames.setEnabled(not running)

        self.chk_extract_video.setEnabled(not running)

        self.chk_limit_frames.setEnabled(not running and self.chk_extract_frames.isChecked())

        self.spin_max_frames.setEnabled(not running and self.chk_extract_frames.isChecked() and self.chk_limit_frames.isChecked())

        self.radio_h5.setEnabled(not running)

        self.radio_txt.setEnabled(not running)

        self.spin_threshold.setEnabled((not running and not self.radio_txt.isChecked()))

        self.txt_path_edit.setEnabled((not running and self.radio_txt.isChecked()))

        self.btn_browse_txt.setEnabled((not running and self.radio_txt.isChecked()))

        self.output_path_edit.setEnabled(not running)

        self.btn_browse_output.setEnabled(not running)

        self.chk_cuda.setEnabled(not running)

        self.chk_dry_run.setEnabled(not running)

        self.chk_report.setEnabled(not running)

        self.chk_commands.setEnabled((not running and self.chk_report.isChecked()))

        self.spin_command_size.setEnabled((not running and self.chk_report.isChecked() and self.chk_commands.isChecked()))

    # ------------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------------

    def run_extraction(self):

        selected_items = self.list_widget.selectedItems()

        if not selected_items:
            QMessageBox.warning(
                self,
                "No videos selected",
                "Select at least one video.",
            )

            return

        extract_frames = self.chk_extract_frames.isChecked()

        extract_video = self.chk_extract_video.isChecked()

        generate_report = self.chk_report.isChecked()

        if not extract_frames and not extract_video and not generate_report:
            QMessageBox.warning(
                self,
                "No action selected",
                "Select Frames, Video, or Generate analysis report.",
            )

            return

        use_txt = self.radio_txt.isChecked()

        txt_path = self.txt_path_edit.text().strip()

        if use_txt:
            if not txt_path:
                QMessageBox.warning(
                    self,
                    "Missing TXT file",
                    "Select a TXT indices file.",
                )

                return

            if not Path(txt_path).exists():
                QMessageBox.warning(
                    self,
                    "Invalid TXT file",
                    f"File does not exist:\n{txt_path}",
                )

                return

        video_paths = [item.text() for item in selected_items]

        config = ExtractionConfig(
            extract_frames=extract_frames,
            extract_video=extract_video,
            limit_frames=(self.chk_limit_frames.isChecked() and extract_frames),
            max_frames=(self.spin_max_frames.value()),
            use_txt=use_txt,
            txt_path=txt_path,
            threshold=(self.spin_threshold.value()),
            output_directory=(self.output_path_edit.text().strip()),
            use_cuda=(self.chk_cuda.isChecked()),
            dry_run=(self.chk_dry_run.isChecked()),
            generate_report=generate_report,
            include_ffmpeg_command=(self.chk_commands.isChecked()),
            command_max_length=(self.spin_command_size.value()),
        )

        self.log_text.clear()

        self.append_log(f"Queued {len(video_paths)} video(s).")

        self.progress_bar.setValue(0)

        self.current_video_label.setText("Starting...")

        self.set_running_state(True)

        self.worker = ExtractionWorker(
            video_paths=video_paths,
            config=config,
        )

        self.worker.log_msg.connect(self.append_log)

        self.worker.video_started.connect(self.on_video_started)

        self.worker.video_finished.connect(self.on_video_finished)

        self.worker.progress_changed.connect(self.progress_bar.setValue)

        self.worker.process_finished.connect(self.on_worker_finished)

        self.worker.process_cancelled.connect(self.on_worker_cancelled)

        self.worker.finished.connect(self.cleanup_worker)

        self.worker.start()

    # ------------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------------

    def cancel_extraction(self):

        if self.worker is None:
            return

        self.append_log("\nCancellation requested...")

        self.btn_cancel.setEnabled(False)

        self.worker.request_cancel()

    # ------------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------------

    def on_video_started(
        self,
        filename: str,
        number: int,
        total: int,
    ):

        self.current_video_label.setText(f"Processing {number}/{total}: {filename}")

    def on_video_finished(
        self,
        filename: str,
        success: bool,
    ):

        status = "SUCCESS" if success else "FAILED"

        self.append_log(f"\n[{status}] {filename}\n")

    def on_worker_finished(
        self,
        success: bool,
    ):

        self.set_running_state(False)

        if success:
            self.progress_bar.setValue(100)

            QMessageBox.information(
                self,
                "Completed",
                ("All selected videos were processed successfully."),
            )

        else:
            QMessageBox.warning(
                self,
                "Completed with errors",
                ("One or more videos failed. Check the log for details."),
            )

    def on_worker_cancelled(self):

        self.set_running_state(False)

        self.current_video_label.setText("Cancelled")

        QMessageBox.information(
            self,
            "Cancelled",
            "Extraction was cancelled.",
        )

    def cleanup_worker(self):

        if self.worker is not None:
            self.worker.deleteLater()

            self.worker = None

    # ------------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------------

    def closeEvent(
        self,
        event,
    ):

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Extraction running",
                ("Cancel the extraction before closing."),
            )

            event.ignore()

            return

        event.accept()


# ============================================================================
# MAIN
# ============================================================================


def main():

    app = QApplication(sys.argv)

    window = MainWindow()

    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
