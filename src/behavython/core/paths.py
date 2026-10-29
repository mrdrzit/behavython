import os
import platform
from pathlib import Path
from behavython.core.defaults import FFMPEG_DOWNLOAD_URLS

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# --- USER ROOT ---
USER_HOME = Path(os.getenv("BEHAVYTHON_HOME", Path.home() / ".behavython"))

# --- USER RUNTIME STRUCTURE ---
USER_LOGS_ROOT = USER_HOME / "logs"
USER_BIN_ROOT = USER_HOME / "bin"
USER_MODELS_ROOT = USER_HOME / "models"


# --- RESOURCES ---
GUI_ROOT = PACKAGE_ROOT / "gui"

GUI_RESOURCES_ROOT = GUI_ROOT / "assets"
GUI_IMAGES_ROOT = GUI_RESOURCES_ROOT / "images"
GUI_UI_ROOT = GUI_RESOURCES_ROOT / "ui"

UI_FILE = GUI_UI_ROOT / "behavython_gui.ui"

LOGO_PATH = GUI_IMAGES_ROOT / "logo.png"
ICON_PATH = GUI_IMAGES_ROOT / "VY.ico"

# --- EXTERNAL URLS ---
# FFMPEG_URL is resolved at import time from the platform-keyed dict in defaults.
# On macOS it will be None — the GUI dependency dialog shows Homebrew instructions instead.

FFMPEG_URL: str | None = FFMPEG_DOWNLOAD_URLS.get(platform.system())
MODELS_URLS = {
    "c57_network_2025_minified": "https://github.com/mrdrzit/behavython/releases/download/models-v2.0/c57_network_2025_minified.zip",
    "cd1_network_2026": "https://github.com/mrdrzit/behavython/releases/download/models-v2.0/cd1_network_2026.zip",
    "roi_network_2026": "https://github.com/mrdrzit/behavython/releases/download/models-v2.0/roi_network_2026.zip",
}
