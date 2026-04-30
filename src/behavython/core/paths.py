import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# --- USER ROOT ---
USER_HOME = Path(os.getenv("BEHAVYTHON_HOME", Path.home() / ".behavython"))

# --- USER RUNTIME STRUCTURE ---
RUNTIME_ROOT = USER_HOME / "runtime"
DATA_ROOT = USER_HOME / "data"

DATA_LOGS_ROOT = DATA_ROOT / "logs"
DATA_HISTORY_ROOT = DATA_ROOT / "history"
DATA_CACHE_ROOT = DATA_ROOT / "cache"

USER_BIN_ROOT = USER_HOME / "bin"
USER_MODELS_ROOT = USER_HOME / "models"


# --- RESOURCES ---
GUI_ROOT = PACKAGE_ROOT / "gui"

GUI_RESOURCES_ROOT = GUI_ROOT / "assets"
GUI_IMAGES_ROOT = GUI_RESOURCES_ROOT / "images"
GUI_UI_ROOT = GUI_RESOURCES_ROOT / "ui"

UI_FILE = GUI_UI_ROOT / "behavython_gui_refactored.ui"

LOGO_PATH = GUI_IMAGES_ROOT / "logo.png"
ICON_PATH = GUI_IMAGES_ROOT / "VY.ico"

# --- EXTERNAL URLS ---
FFMPEG_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
MODELS_URLS = {
    "c57_network_2025_minified": "https://github.com/mrdrzit/behavython/releases/download/models-v1.0/c57_network_2025_minified.zip",
    "roi_network": "https://github.com/mrdrzit/behavython/releases/download/models-v1.0/roi_network.zip",
}
