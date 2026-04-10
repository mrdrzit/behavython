import os
import shutil
import platform
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QMessageBox
from PySide6.QtCore import Qt
from matplotlib.path import Path
from behavython.services.downloader import downloadWorker
from behavython.core.paths import USER_BIN_ROOT, USER_MODELS_ROOT


class dependencyDownloadDialog(QDialog):
    def __init__(self, target: str, url: str, parent=None):
        super().__init__(parent)
        self.target = target
        self.url = url
        self.setWindowTitle(f"Downloading Dependency: {target.upper()}")
        self.setFixedSize(400, 150)
        self.setWindowModality(Qt.ApplicationModal)

        layout = QVBoxLayout(self)

        self.status_label = QLabel(f"Preparing to download {target}...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_download)
        layout.addWidget(self.cancel_button)

        self.worker = downloadWorker(self.target, self.url)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_download_finished)

    def start_download(self):
        self.cancel_button.setEnabled(True)
        self.worker.start()
        self.exec()

    def cancel_download(self):
        self.worker.cancel()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Cancelling...")

    def on_download_finished(self, success: bool, message: str):
        if success:
            self.accept()
        else:
            QMessageBox.critical(self, "Download Error", f"Failed to download {self.target}:\n{message}")
            self.reject()


def get_os_name() -> str:
    return platform.system()


def is_ffmpeg_installed() -> bool:
    binary_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    # 1. Check local Behavython bin folder (for Windows downloads)
    local_bin = USER_BIN_ROOT / binary_name
    if local_bin.exists():
        return True

    # 2. Check system PATH (for macOS/Linux/Windows PATH)
    if shutil.which("ffmpeg"):
        return True

    return False


def get_unix_install_instructions() -> str:
    sys_name = get_os_name()
    if sys_name == "Darwin":
        return (
            "On macOS, the easiest way to install FFmpeg is via Homebrew:\n\n"
            "    brew install ffmpeg\n\n"
            "Please open your Terminal, run this command, and restart Behavython."
        )
    elif sys_name == "Linux":
        return (
            "On Linux, you can install FFmpeg via your package manager.\n"
            "For Ubuntu/Debian:\n\n"
            "    sudo apt update && sudo apt install ffmpeg\n\n"
            "Please open your Terminal, run this command, and restart Behavython."
        )
    return "Please install FFmpeg and ensure it is available in your system's PATH."


def is_model_installed(model_name: str) -> bool:
    model_dir = USER_MODELS_ROOT / model_name
    return model_dir.exists() and any(model_dir.iterdir())


def get_model_path(model_name: str) -> Path:
    """Returns the absolute path to the model's root directory."""
    return USER_MODELS_ROOT / model_name
