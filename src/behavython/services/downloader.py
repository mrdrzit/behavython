import os
import shutil
import tarfile
import zipfile
import platform
import requests
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from behavython.core.paths import USER_BIN_ROOT, USER_MODELS_ROOT
from behavython.core.defaults import FFPROBE_MACOS_URL


# Binary names differ by platform
_IS_WINDOWS = os.name == "nt"
_IS_MACOS   = platform.system() == "Darwin"
_FFMPEG_BIN  = "ffmpeg.exe"  if _IS_WINDOWS else "ffmpeg"
_FFPROBE_BIN = "ffprobe.exe" if _IS_WINDOWS else "ffprobe"


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
        self.status.emit(f"Downloading from {url.split('/')[2]}...")
        response = requests.get(url, stream=True, timeout=30)
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

    def _extract_flat_zip(self, archive_path: Path, binary_name: str, dest_path: Path) -> bool:
        """
        Extracts a single binary from a flat zip (evermeet.cx style).
        The archive contains just the binary at the root — no bin/ subdirectory.
        Returns True if the binary was found and extracted.
        """
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                # The binary is at the root and named exactly (e.g. "ffmpeg")
                if Path(member.filename).name == binary_name and not member.filename.endswith("/"):
                    with zip_ref.open(member) as src, open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    return True
        return False

    def _download_and_extract_ffmpeg(self):
        USER_BIN_ROOT.mkdir(parents=True, exist_ok=True)

        if _IS_MACOS:
            self._download_and_extract_ffmpeg_macos()
        else:
            self._download_and_extract_ffmpeg_unix_or_windows()

    def _download_and_extract_ffmpeg_macos(self):
        """
        macOS: evermeet.cx provides ffmpeg and ffprobe as two separate flat zips.
        Each zip contains a single binary at the root.
        """
        for binary_name, url in [(_FFMPEG_BIN, self.url), (_FFPROBE_BIN, FFPROBE_MACOS_URL)]:
            if self._is_cancelled:
                return

            archive_path = USER_BIN_ROOT / f"{binary_name}_temp.zip"
            self._download_file(url, archive_path)

            if self._is_cancelled:
                archive_path.unlink(missing_ok=True)
                return

            self.status.emit(f"Extracting {binary_name}...")
            dest = USER_BIN_ROOT / binary_name
            found = self._extract_flat_zip(archive_path, binary_name, dest)
            archive_path.unlink(missing_ok=True)

            if not found:
                raise FileNotFoundError(f"Could not find '{binary_name}' inside the downloaded archive.")

            # Ensure the binary is executable
            dest.chmod(dest.stat().st_mode | 0o111)

            self.progress.emit(100 if binary_name == _FFPROBE_BIN else 50)

    def _download_and_extract_ffmpeg_unix_or_windows(self):
        """
        Windows + Linux: BtbN provides a single archive containing both binaries
        inside a `bin/` subdirectory.
        Windows: zip  (ffmpeg.exe, ffprobe.exe)
        Linux:   tar.xz (ffmpeg, ffprobe)
        """
        is_tar = self.url.endswith(".tar.xz")
        archive_ext = ".tar.xz" if is_tar else ".zip"
        archive_path = USER_BIN_ROOT / f"ffmpeg_temp{archive_ext}"

        self._download_file(self.url, archive_path)
        if self._is_cancelled:
            archive_path.unlink(missing_ok=True)
            return

        self.status.emit("Extracting FFmpeg...")
        self.progress.emit(0)

        extract_dir = USER_BIN_ROOT / "ffmpeg_extracted"
        extract_dir_resolved = extract_dir.resolve()

        if is_tar:
            with tarfile.open(archive_path, "r:xz") as tar:
                for member in tar.getmembers():
                    target_path = (extract_dir / member.name).resolve()
                    if not str(target_path).startswith(str(extract_dir_resolved)):
                        raise RuntimeError(f"Security hazard: path traversal attempt in {member.name}")
                tar.extractall(extract_dir)
        else:
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                for member in zip_ref.infolist():
                    target_path = (extract_dir / member.filename).resolve()
                    if not target_path.is_relative_to(extract_dir_resolved):
                        raise RuntimeError(f"Security hazard: Zip Slip attempt detected in file {member.filename}")
                    zip_ref.extract(member, extract_dir)

        bin_dir = next(extract_dir.rglob("bin"), None)

        if bin_dir is None:
            shutil.rmtree(extract_dir, ignore_errors=True)
            archive_path.unlink(missing_ok=True)
            raise FileNotFoundError("Could not locate the 'bin' directory within the extracted FFmpeg archive.")

        for binary_name in [_FFMPEG_BIN, _FFPROBE_BIN]:
            src_bin = bin_dir / binary_name
            if src_bin.exists():
                dest = USER_BIN_ROOT / binary_name
                shutil.move(str(src_bin), str(dest))
                if not _IS_WINDOWS:
                    dest.chmod(dest.stat().st_mode | 0o111)

        self.status.emit("Cleaning up...")
        archive_path.unlink(missing_ok=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

    def _download_and_extract_model(self):
        USER_MODELS_ROOT.mkdir(parents=True, exist_ok=True)
        zip_path = USER_MODELS_ROOT / f"{self.target}.zip"

        self._download_file(self.url, zip_path)
        if self._is_cancelled:
            return

        self.status.emit(f"Extracting {self.target}...")
        self.progress.emit(0)

        model_dir = USER_MODELS_ROOT / self.target
        model_dir.mkdir(exist_ok=True)
        model_dir_resolved = model_dir.resolve()

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.infolist():
                target_path = (model_dir / member.filename).resolve()

                if not target_path.is_relative_to(model_dir_resolved):
                    raise RuntimeError(f"Security hazard: Zip Slip attempt detected in file {member.filename}")

                zip_ref.extract(member, model_dir)

        self.status.emit("Cleaning up...")
        zip_path.unlink(missing_ok=True)
