import cv2
import subprocess
import logging
from pathlib import Path
from behavython.core.utils import get_ffmpeg_path, get_ffprobe_path

logger = logging.getLogger(__name__)

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

    @classmethod
    def has_nvenc_support(cls) -> bool:
        """
        Dynamically tests if the system has an Nvidia GPU and drivers ready for NVENC.
        """
        if cls._nvenc_available is not None:
            return cls._nvenc_available
            
        ffmpeg = get_ffmpeg_path()
        cmd = [
            ffmpeg, "-y", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        try:
            result = subprocess.run(cmd, capture_output=True)
            cls._nvenc_available = (result.returncode == 0)
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
            ffprobe, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            codec = result.stdout.strip().lower()
            # If it's not h264 or hevc (h265), flag it as problematic
            return codec not in ["h264", "hevc"]
        except Exception:
            return True # Flag as problematic if ffprobe fails

    @classmethod
    def crop_video(cls, video_path: Path, output_path: Path, crop_data: dict) -> bool:
        """
        Uses FFmpeg with Nvidia hardware acceleration to rotate and crop the video.
        crop_data expects: {'x': int, 'y': int, 'width': int, 'height': int, 'rotation': float}
        """
        x = crop_data.get("x", 0)
        y = crop_data.get("y", 0)
        w = crop_data.get("width", 0)
        h = crop_data.get("height", 0)
        angle_deg = crop_data.get("rotation", 0.0)
        
        # Build the FFmpeg filter chain
        filters = []
        if angle_deg != 0.0:
            # Expand the canvas to fit the rotated video, fill empty space with black
            # rotate=angle*PI/180:ow=hypot(iw,ih):oh=ow:c=black
            filters.append(f"rotate={angle_deg}*PI/180:ow='hypot(iw,ih)':oh=ow:c=black")
            
        filters.append(f"crop={w}:{h}:{x}:{y}")
        vf_string = ",".join(filters)
        
        ffmpeg = get_ffmpeg_path()
        cmd = [ffmpeg, "-y"]
        
        if cls.has_nvenc_support():
            # GPU Accelerated path
            cmd.extend([
                "-threads", "0",
                "-hwaccel", "auto",
                "-i", str(video_path),
                "-vf", vf_string,
                "-c:v", "h264_nvenc",
                "-preset", "p2",
                "-cq", "23",
                "-b:v", "0"
            ])
        else:
            # CPU Fallback path
            cmd.extend([
                "-i", str(video_path),
                "-vf", vf_string,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23"
            ])
            
        cmd.extend([
            "-c:a", "copy",
            str(output_path)
        ])
        
        logger.info(f"Running FFmpeg crop: {' '.join(cmd)}")
        
        try:
            # We remove capture_output so FFmpeg natively streams its progress to the terminal
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with error:\n{e}")
            return False

def run_batch_crop(request: dict, progress=None, log=None, warning=None) -> dict:
    """
    Worker task that runs the FFmpeg batch crop pipeline and updates the GUI progress bar.
    """
    import json
    from pathlib import Path
    
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
    
    for i, vid in enumerate(videos_to_process):
        video_path = Path(vid)
        if not video_path.exists():
            if log:
                log.emit("dlc", f"File not found: {video_path.name}")
            continue
            
        data = project_data[vid]
        c = data["coordinates"]
        
        # Create output directory
        output_dir = video_path.parent / "cropped"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / video_path.name
        
        if log:
            log.emit("dlc", f"[{i+1}/{total}] Cropping {video_path.name}...")
        
        # Run FFmpeg (prints to terminal natively)
        success = VideoService.crop_video(video_path, output_path, c)
        
        if success:
            data["video_cropped"] = True
            processed += 1
            # Save progress incrementally to the JSON file
            with open(project_path, "w") as f:
                json.dump(project_data, f, indent=4)
            if log:
                log.emit("dlc", f"Successfully saved to {output_path.name}")
        else:
            if log:
                log.emit("dlc", f"FFmpeg failed to crop {video_path.name}")
            
        # Update UI progress bar
        if progress:
            prog_val = int(((i + 1) / total) * 100)
            progress.emit(prog_val)
        
    return {"kind": "batch_crop", "processed": processed, "total": total}

def run_standardize_videos(request: dict, progress=None, log=None, warning=None) -> dict:
    """
    Worker task that converts all videos in a folder to standardized H.264 (CFR, NVENC).
    """
    import shutil
    from pathlib import Path
    from behavython.services.video_service import VideoService
    
    videos = request["videos"]
    total = len(videos)
    processed = 0
    ffmpeg = get_ffmpeg_path()
    
    for i, vid in enumerate(videos):
        vid_path = Path(vid)
        if log:
            log.emit("dlc", f"[{i+1}/{total}] Standardizing {vid_path.name}...")
            
        # Create a temp output file
        temp_out = vid_path.parent / f"temp_std_{vid_path.name}"
        
        cmd = [ffmpeg, "-y", "-i", str(vid_path)]
        
        if VideoService.has_nvenc_support():
            cmd.extend(["-c:v", "h264_nvenc", "-preset", "p2", "-cq", "23", "-b:v", "0"])
        else:
            cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
            
        cmd.extend(["-c:a", "copy", str(temp_out)])
        
        try:
            # We want to see ffmpeg progress natively in terminal
            subprocess.run(cmd, check=True)
            
            # Back up original to unwanted_files to keep project clean
            unwanted_dir = vid_path.parent / "unwanted_files"
            unwanted_dir.mkdir(exist_ok=True)
            original_backup = unwanted_dir / f"original_{vid_path.name}"
            shutil.move(str(vid_path), str(original_backup))
            
            # Rename temp to original name so we don't break downstream workflows
            shutil.move(str(temp_out), str(vid_path))
            
            processed += 1
            if log:
                log.emit("dlc", f"Successfully standardized {vid_path.name}")
        except Exception as e:
            if log:
                log.emit("dlc", f"Failed to standardize {vid_path.name}: {e}")
                
        if progress:
            progress.emit(int(((i+1)/total)*100))
            
    return {"kind": "standardize", "processed": processed, "total": total}
