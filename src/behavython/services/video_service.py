import cv2
import subprocess
import logging
import threading
from pathlib import Path
from behavython.core.utils import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger("behavython.console")


class VideoService:
    @staticmethod
    def extract_preview_frame(video_path: Path, output_path: Path) -> bool:
        """
        Extracts the first frame of a video using OpenCV and saves it to output_path.
        OpenCV is completely synchronous and very fast for single frame extraction.
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return False

            ret, frame = cap.read()
            cap.release()

            if ret:
                cv2.imwrite(str(output_path), frame)
                return True
            return False
        except Exception as e:
            logger.error(f"Error extracting frame: {e}")
            return False

    _nvenc_available = None
    _nvenc_lock = threading.Lock()

    @classmethod
    def has_nvenc_support(cls) -> bool:
        """
        Dynamically tests if the system has an Nvidia GPU and drivers ready for NVENC.
        """
        if cls._nvenc_available is not None:
            return cls._nvenc_available

        with cls._nvenc_lock:
            # Double-check inside the lock to prevent race conditions
            if cls._nvenc_available is not None:
                return cls._nvenc_available

            ffmpeg = get_ffmpeg_path()
            cmd = [ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1", "-c:v", "h264_nvenc", "-f", "null", "-"]
            try:
                result = subprocess.run(cmd, capture_output=True)
                cls._nvenc_available = result.returncode == 0
            except Exception:
                cls._nvenc_available = False

            if cls._nvenc_available:
                logger.info("Nvidia NVENC acceleration is fully supported and ready.")
            else:
                logger.warning("NVENC acceleration not available. Falling back to libx264 CPU encoding.")

            return cls._nvenc_available

    @staticmethod
    def is_legacy_codec(video_path: Path) -> bool:
        """
        Uses ffprobe to detect if the video uses a legacy codec or is not h264/hevc.
        """
        ffprobe = get_ffprobe_path()
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            codec = result.stdout.strip().lower()
            # If it's not h264 or hevc (h265), flag it as problematic
            return codec not in ["h264", "hevc"]
        except Exception:
            return True  # Flag as problematic if ffprobe fails

    @staticmethod
    def run_ffmpeg_with_tqdm(cmd: list, video_path: Path, desc: str, position: int = None) -> bool:
        """
        Runs FFmpeg and uses tqdm to display a progress bar based on frame count.
        """
        from tqdm import tqdm
        import cv2
        import re
        from collections import deque

        # Get total frames for the progress bar
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Remove manual console flags so tqdm can parse stderr cleanly
        cmd_clean = [c for c in cmd if c not in ["-hide_banner", "-loglevel", "error", "-stats"]]

        frame_pattern = re.compile(r"frame=\s*(\d+)")
        last_errors = deque(maxlen=10)

        try:
            process = subprocess.Popen(
                cmd_clean, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, universal_newlines=True, encoding="utf-8", errors="replace"
            )

            # leave=False hides the bar when finished to keep console clean
            with tqdm(total=total_frames, desc=desc[:30], unit="frames", leave=False, position=position) as pbar:
                last_frame = 0
                for line in process.stderr:
                    match = frame_pattern.search(line)
                    if match:
                        current_frame = int(match.group(1))
                        if current_frame > last_frame:
                            pbar.update(current_frame - last_frame)
                            last_frame = current_frame
                    elif line.strip():
                        last_errors.append(line.strip())

            process.wait()
            if process.returncode != 0:
                error_msg = "\n".join(last_errors)
                logger.error(f"FFmpeg failed: {error_msg}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error running FFmpeg with tqdm: {e}")
            return False

    @classmethod
    def crop_video(cls, video_path: Path, output_path: Path, crop_data: dict, position: int = None) -> bool:
        """
        Uses FFmpeg with Nvidia hardware acceleration to rotate and crop the video.
        crop_data expects: {'x': int, 'y': int, 'width': int, 'height': int, 'rotation': float}
        """
        x = int(crop_data.get("x", 0))
        y = int(crop_data.get("y", 0))
        w = int(crop_data.get("width", 0))
        h = int(crop_data.get("height", 0))
        angle_deg = crop_data.get("rotation", 0.0)

        # Force even dimensions (strictly required by H.264 / yuv420p)
        if w % 2 != 0:
            w -= 1
        if h % 2 != 0:
            h -= 1

        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]

        use_nvenc = cls.has_nvenc_support()

        # NVENC has a hard limit on minimum frame sizes (often 128x128 or 144x144).
        # CPU encoding a video this small is instantaneous anyway.
        if use_nvenc and (w < 144 or h < 144):
            use_nvenc = False

        if use_nvenc:
            # HYBRID PIPELINE (CPU Filters for rotation/crop -> GPU Encode)
            filters = []
            if angle_deg != 0.0:
                filters.append(f"rotate={angle_deg}*PI/180:ow='hypot(iw,ih)':oh=ow:c=black")
            filters.append(f"crop={w}:{h}:{x}:{y}")
            vf_string = ",".join(filters)
            cmd.extend(
                [
                    "-threads",
                    "0",
                    "-hwaccel",
                    "auto",
                    "-i",
                    str(video_path),
                    "-vf",
                    vf_string,
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p2",
                    "-cq",
                    "23",
                    "-b:v",
                    "0",
                ]
            )
        else:
            # CPU Fallback path
            filters = []
            if angle_deg != 0.0:
                filters.append(f"rotate={angle_deg}*PI/180:ow='hypot(iw,ih)':oh=ow:c=black")
            filters.append(f"crop={w}:{h}:{x}:{y}")
            vf_string = ",".join(filters)

            cmd.extend(["-i", str(video_path), "-vf", vf_string, "-c:v", "libx264", "-preset", "fast", "-crf", "23"])

        cmd.extend(["-c:a", "copy", str(output_path)])

        return cls.run_ffmpeg_with_tqdm(cmd, video_path, f"Cropping {video_path.name}", position=position)


def run_batch_crop(request: dict, progress=None, log=None, warning=None) -> dict:
    """
    Worker task that runs the FFmpeg batch crop pipeline and updates the GUI progress bar.
    Runs 2 videos in parallel using ThreadPoolExecutor.
    """
    import json
    import threading
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed

    project_data = request["project_data"]
    project_path = Path(request["project_path"])

    # Filter videos that are ready and not yet cropped
    videos_to_process = [vid for vid, data in project_data.items() if data.get("coordinates_set") and not data.get("video_cropped")]

    if not videos_to_process:
        if log:
            log.emit("dlc", "No new videos to crop (or no coordinates set).")
        return {"kind": "batch_crop", "processed": 0, "total": 0}

    total = len(videos_to_process)
    processed = 0
    io_lock = threading.Lock()
    logger.info(f"Starting to crop {total} videos.")

    def process_single_video(vid, index):
        video_path = Path(vid)
        if not video_path.exists():
            with io_lock:
                if log:
                    log.emit("dlc", f"File not found: {video_path.name}")
            return False

        data = project_data[vid]
        c = data["coordinates"]

        # Create output directory
        output_dir = video_path.parent / "cropped"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / video_path.name

        with io_lock:
            if log:
                log.emit("dlc", f"[{index + 1}/{total}] Cropping {video_path.name}...")

        # Run FFmpeg with tqdm progress bar
        success = VideoService.crop_video(video_path, output_path, c, position=index % 2)

        if success:
            with io_lock:
                project_data[vid]["video_cropped"] = True
                # Save progress incrementally to the JSON file
                with open(project_path, "w") as f:
                    json.dump(project_data, f, indent=4)
                if log:
                    log.emit("dlc", f"Successfully saved to {output_path.name}")
        else:
            with io_lock:
                if log:
                    log.emit("dlc", f"FFmpeg failed to crop {video_path.name}")

        return success

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit all tasks
        futures = {executor.submit(process_single_video, vid, i): vid for i, vid in enumerate(videos_to_process)}

        for i, future in enumerate(as_completed(futures)):
            if future.result():
                processed += 1
            # Update UI progress bar
            if progress:
                prog_val = int(((i + 1) / total) * 100)
                progress.emit(prog_val)

    print("\r", end="", flush=True)  # Reset cursor to left margin without adding a newline
    logger.info(f"Finished cropping {processed} videos.")
    return {"kind": "batch_crop", "processed": processed, "total": total}


def run_standardize_videos(request: dict, progress=None, log=None, warning=None) -> dict:
    """
    Worker task that converts all videos in a folder to standardized H.264 (CFR, NVENC).
    Runs 2 videos in parallel.
    """
    import shutil
    import threading
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from behavython.services.video_service import VideoService

    videos = request["videos"]
    total = len(videos)
    processed = 0
    ffmpeg = get_ffmpeg_path()
    io_lock = threading.Lock()
    logger.info(f"Starting to normalize {total} videos.")

    def standardize_single(vid, index):
        vid_path = Path(vid)
        with io_lock:
            if log:
                log.emit("dlc", f"[{index + 1}/{total}] Standardizing {vid_path.name}...")

        # Create a temp output file
        temp_out = vid_path.parent / f"temp_std_{vid_path.name}"

        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(vid_path)]

        if VideoService.has_nvenc_support():
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p2", "-cq", "23", "-b:v", "0"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])

        # Force Constant Frame Rate and standard pixel format
        cmd.extend(["-vsync", "cfr", "-pix_fmt", "yuv420p"])
        cmd.extend(["-c:a", "copy", str(temp_out)])

        success = VideoService.run_ffmpeg_with_tqdm(cmd, vid_path, f"Standardizing {vid_path.name}", position=index % 2)

        # Validation: ensure output exists and is not empty
        if not success or not temp_out.exists() or temp_out.stat().st_size == 0:
            with io_lock:
                if log:
                    log.emit("dlc", f"Failed to standardize {vid_path.name}")
            if temp_out.exists():
                temp_out.unlink()  # Cleanup failed temp file
            return False

        with io_lock:
            # Back up original to unwanted_files to keep project clean
            unwanted_dir = vid_path.parent / "unwanted_files"
            unwanted_dir.mkdir(exist_ok=True)
            original_backup = unwanted_dir / f"original_{vid_path.name}"
            shutil.move(str(vid_path), str(original_backup))

            # Rename temp to original name so we don't break downstream workflows
            shutil.move(str(temp_out), str(vid_path))

            if log:
                log.emit("dlc", f"Successfully standardized {vid_path.name}")
        return True

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(standardize_single, vid, i): vid for i, vid in enumerate(videos)}
        for i, future in enumerate(as_completed(futures)):
            if future.result():
                processed += 1
            if progress:
                progress.emit(int(((i + 1) / total) * 100))

    print("\r", end="", flush=True)  # Reset cursor to left margin without adding a newline
    logger.info(f"Finished normalizing {processed} videos.")
    return {"kind": "standardize", "processed": processed, "total": total}
