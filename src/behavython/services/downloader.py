import shutil
import zipfile
import requests
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from behavython.core.paths import USER_BIN_ROOT, USER_MODELS_ROOT


class downloadWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(bool, str)

    def __init__(self, target: str, url: str):
        super().__init__()
        self.target = target
        self.url = url
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            if self.target == "ffmpeg":
                self._download_and_extract_ffmpeg()
            else:
                self._download_and_extract_model()

            if not self._is_cancelled:
                self.finished.emit(True, "Download complete.")
        except Exception as e:
            self.finished.emit(False, str(e))

    def _download_file(self, url: str, dest_path: Path):
        self.status.emit(f"Downloading {self.target}...")
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192
        downloaded_size = 0

        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if self._is_cancelled:
                    raise InterruptedError("Download cancelled by user.")
                f.write(chunk)
                downloaded_size += len(chunk)
                if total_size > 0:
                    percent = int((downloaded_size / total_size) * 100)
                    self.progress.emit(percent)

    def _download_and_extract_ffmpeg(self):
        USER_BIN_ROOT.mkdir(parents=True, exist_ok=True)
        zip_path = USER_BIN_ROOT / "ffmpeg_temp.zip"

        self._download_file(self.url, zip_path)
        if self._is_cancelled:
            return

        self.status.emit("Extracting FFmpeg...")
        self.progress.emit(0)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            extract_dir = USER_BIN_ROOT / "ffmpeg_extracted"
            zip_ref.extractall(extract_dir)

            bin_dir = next(extract_dir.rglob("bin"))

            for exe in ["ffmpeg.exe", "ffprobe.exe"]:
                src_exe = bin_dir / exe
                if src_exe.exists():
                    shutil.move(str(src_exe), str(USER_BIN_ROOT / exe))

        self.status.emit("Cleaning up...")
        zip_path.unlink()
        shutil.rmtree(extract_dir)

    def _download_and_extract_model(self):
        USER_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        zip_path = USER_MODELS_ROOT / f"{self.target}.zip"

        self._download_file(self.url, zip_path)
        if self._is_cancelled:
            return

        self.status.emit(f"Extracting {self.target}...")
        self.progress.emit(0)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            model_dir = USER_MODELS_ROOT / self.target
            model_dir.mkdir(exist_ok=True)
            zip_ref.extractall(model_dir)

        self.status.emit("Cleaning up...")
        zip_path.unlink()
