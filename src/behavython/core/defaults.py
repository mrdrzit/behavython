# ==========================================
# APPLICATION & GUI SETTINGS
# ==========================================
APP_NAME = "Behavython"
MAIN_WINDOW_TITLE = "Behavython"
LOGO_WIDGET_NAME = "behavython_logo"

# ==========================================
# FFMPEG DOWNLOAD URLS
# ==========================================
# Windows + Linux: BtbN automated GitHub Actions builds.
#   Both ffmpeg and ffprobe are bundled in a single archive.
#   Archive structure: <root>/<name>/bin/ffmpeg[.exe]
#
# macOS: evermeet.cx static builds (John Van Sickle).
#   Third-party but widely trusted; FFmpeg's own download page references it.
#   ffmpeg and ffprobe are distributed as SEPARATE zip archives.
#   Archive structure: flat zip containing just the binary at the root.
#   Using the un-versioned URLs so users always get the latest build.
FFMPEG_DOWNLOAD_URLS: dict[str, str | None] = {
    "Windows": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip",
    "Linux": "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz",
    "Darwin": "https://evermeet.cx/ffmpeg/ffmpeg.zip",
}

# macOS only: separate ffprobe download (evermeet.cx distributes the two binaries separately)
FFPROBE_MACOS_URL: str = "https://evermeet.cx/ffprobe/ffprobe.zip"

SCROLLBAR_STYLE = """
/* =========================================
   VERTICAL SCROLLBAR
   ========================================= */
QScrollBar:vertical {
    border: none;
    background: #2b2b2b;
    width: 12px;
    margin: 0px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background: #5c5c5c;
    min-height: 30px;
    border-radius: 6px;
}

QScrollBar::handle:vertical:hover {
    background: #7a7a7a;
}

QScrollBar::handle:vertical:pressed {
    background: #3a539b;
}

QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical {
    border: none;
    background: none;
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* =========================================
   HORIZONTAL SCROLLBAR
   ========================================= */
QScrollBar:horizontal {
    border: none;
    background: #2b2b2b;
    height: 12px;
    margin: 0px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background: #5c5c5c;
    min-width: 30px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal:hover {
    background: #7a7a7a;
}

QScrollBar::handle:horizontal:pressed {
    background: #3a539b;
}

QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal {
    border: none;
    background: none;
    width: 0px;
}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    background: none;
}
"""

DEFAULT_STYLE = (
    """
QWidget {
    background-color: #343434;
    color: #FFFFFF;
}
"""
    + "\n"
    + SCROLLBAR_STYLE
)

DEBUG_STYLE = (
    """
/* Global colors only */
QWidget {
    background-color: #1E1E1E;
    color: #CCCCCC;
}

/* Text areas */
QTextEdit, QPlainTextEdit, QLineEdit {
    background-color: #121212;
    color: #CE9178;
}

/* Buttons */
QPushButton {
    background-color: #2D2D2D;
    color: #d97706;
}

QPushButton:hover {
    background-color: #d97706;
    color: #1E1E1E;
}

QPushButton:pressed {
    background-color: #b45309;
    color: #FFFFFF;
}

/* Progress bar */
QProgressBar {
    background-color: #252526;
    color: #FFFFFF;
}

QProgressBar::chunk {
    background-color: #d97706;
}

/* ComboBox */
QComboBox {
    background-color: #252526;
    color: #D4D4D4;
}

QComboBox QAbstractItemView {
    background-color: #1E1E1E;
    color: #CE9178;
    selection-background-color: #d97706;
    selection-color: #1E1E1E;
}

/* Checkbox */
QCheckBox {
    color: #D4D4D4;
}

QCheckBox::indicator:checked {
    background-color: #d97706;
}
"""
    + "\n"
    + SCROLLBAR_STYLE
)

# Validation checkbox states
VALIDATION_CHECKBOX_ACTIVE = """
QCheckBox { color: #FFFFFF; }
QCheckBox::indicator { width:15px; height:15px; background-color:#606060; border-radius:4px; }
QCheckBox::indicator:checked { background-color:#A21F27; }
"""

# Faded state
VALIDATION_CHECKBOX_FADED = """
QCheckBox { color: #808080; }
QCheckBox::indicator { width:15px; height:15px; background-color:#404040; border-radius:4px; }
QCheckBox::indicator:checked { background-color:#5a1116; }
"""

# ==========================================
# DEFAULT GUI STATES
# ==========================================
DEFAULT_EXPERIMENT_TYPE_INDEX = 0
DEFAULT_ALGO_TYPE_INDEX = 0
DEFAULT_ANIMAL_INDEX = 0
DEFAULT_FIG_MAX_SIZE_INDEX = 1

DEFAULT_PLOT_ENABLED = True
DEFAULT_CROP_VIDEO = False

# ==========================================
# DEFAULT ANALYSIS PARAMETERS
# ==========================================
DEFAULT_ARENA_WIDTH = 30
DEFAULT_ARENA_HEIGHT = 30
DEFAULT_FRAMES_PER_SECOND = 30
DEFAULT_TASK_DURATION_SECONDS = 300
DEFAULT_TRIM_AMOUNT_SECONDS = 0
MIN_EVENT_FRAMES = 3
LINK_WINDOW_S = 1.0
ANGLE_CONE_ENTER = 45.0
ANGLE_CONE_EXIT = 60.0
ROI_DELTA_LOOKBACK_FRAMES = 3
APPROACH_THRESHOLD_BL = 0.1
LOCOMOTION_THRESHOLD_BL = 0.02

# Per-frame displacement (cm) below which movement is treated as tracking jitter
MOVEMENT_THRESHOLD_CM_BY_ANIMAL = {
    "mouse": 0.0267,  # C57BL/6
    "rat": 0.0667,  # Wistar
}

DEFAULT_ANALYSIS_PARAMETERS = {
    "frames_per_second": 30,
    "arena_width": 30,
    "arena_height": 30,
    "task_duration": 300,
    "trim_amount": 0,
}

# ==========================================
# EXPERIMENT DEFINITIONS
# ==========================================
EXPERIMENT_TYPES = [
    "social_recognition",
    "social_discrimination",
    "object_discrimination",
]

ROI_COUNT_BY_EXPERIMENT = {
    "social_recognition": 1,
    "social_discrimination": 2,
    "object_discrimination": 2,
}

MAZE_EXPERIMENT_TYPES = {
    "open_field",
    "elevated_plus_maze",
}

SOCIAL_EXPERIMENT_TYPES = {
    "social_recognition",
    "social_discrimination",
}

# ==========================================
# BIOLOGICAL & TRACKING MODELS
# ==========================================
CANONICAL_BODYPARTS = [
    "nose",
    "left_ear",
    "right_ear",
    "center",
    "tail",
]

BODYPART_MAPPING = {
    "focinho": "nose",
    "orelhae": "left_ear",
    "orelhad": "right_ear",
    "centro": "center",
    "rabo": "tail",
}

CANONICAL_SKELETON = [
    ("nose", "center"),
    ("left_ear", "center"),
    ("right_ear", "center"),
    ("center", "tail"),
]

# ==========================================
# FILE & SYSTEM CONFIGURATION
# ==========================================
SESSION_LOGS_TO_KEEP = 20  # one log file per app or CLI session in ~/.behavython/logs

# GUI text box that shows the INFO and higher messages of each module. Keys are logger names (module paths);
# a key also covers its submodules. Modules not listed only log to the session file and the terminal.
GUI_LOG_PANELS = {
    "behavython.pipeline.workflow": "resume",
    "behavython.pipeline.export": "resume",
    "behavython.pipeline.models": "resume",  # per-animal issues from Animal.log
    "behavython.pipeline.plugins.dlc": "dlc",
    "behavython.services.video_service": "dlc",
}

ANALYSIS_REQUIRED_SUFFIXES = {
    "image": (".png", ".jpg", ".jpeg", ".tiff"),
    "video": (".mp4", ".avi", ".mov"),
    "position": ("filtered.csv",),
    "skeleton": ("filtered_skeleton.csv",),
    "social_recognition_rois": ("_roi.csv",),
    "social_discrimination_rois": ("roil.csv", "roir.csv"),
    "maze_rois": (".json",),
}

# Default pose networks, in the order of the GUI default_network_combobox. Keys are the CLI --network choices.
DEFAULT_POSE_NETWORKS = {
    "c57": "c57_network_2025_minified",
    "cd1": "cd1_network_2026",
}

# ==========================================
# AUTOMATIC ROI CREATION (find_cup_circle network)
# ==========================================
# The bundled network is trained on the CD1 and C57 social recognition setups of our lab.
# Other setups need their own DLC network with these four bodyparts on the cup circle.
ROI_NETWORK_MODEL_NAME = "roi_network_2026"
ROI_NETWORK_BODYPARTS = ("top", "right", "bottom", "left")
ROI_FRAMES_PER_VIDEO = 5
ROI_SKIP_START_FRACTION = 0.1  # the animal is being placed in the arena
ROI_FRAME_DEVIATION_WARNING_PX = 10.0  # a sampled frame whose circle center or diameter differs this much from the median
ROI_DIAMETER_OUTLIER_FRACTION = 0.1  # a video whose diameter differs this much from the folder median
ROI_CHECK_FOLDER = "roi_check"

# ==========================================
# CORE COLOR PALETTE
# ==========================================
HEX_ORANGERED = "#FF4500"
BGR_ORANGERED = (0, 69, 255)

HEX_CYAN = "#00FFFF"
BGR_CYAN = (255, 255, 0)

HEX_MAGENTA = "#FF00FF"
BGR_MAGENTA = (255, 0, 255)

# ==========================================
# PLOTTING & VISUALIZATION STYLES (MATPLOTLIB)
# ==========================================
PLOT_COLORMAP = "inferno"
# KDE evaluation grid per axis for the exploration map; seaborn's default is 200 and is 4x slower with no visible change.
PLOT_KDE_GRIDSIZE = 100
PLOT_TRAJECTORY_COLOR = HEX_ORANGERED
PLOT_TRAJECTORY_LINEWIDTH = 1.5
PLOT_SCATTER_SIZE = 6

ZONE_COLOR_MAP = {
    "top_open": "red",
    "right_closed": "blue",
    "bottom_open": "green",
    "left_closed": "yellow",
    "center": "purple",
    "none": "magenta",
    "missing": "magenta",
    "outlier": "magenta",
}

ZONE_FALLBACK_COLORS = ["cyan", "lime", "orange", "pink", "teal", "coral", "gold", "white", "brown"]
PLOT_GEOMETRY_COLORS = ["red", "blue", "green", "yellow", "purple", "cyan", "magenta", "orange", "lime"]
ANIMATION_POLY_COLORS = ["lightblue", "lightgreen", "lightcoral", "moccasin", "lightgrey", "pink"]

# ==========================================
# OPENCV VIDEO ANIMATION STYLES (BGR Format)
# ==========================================
CV2_TRAJECTORY_COLOR = BGR_ORANGERED
CV2_CROSSING_COLOR = (255, 255, 255)
CV2_TEXT_COLOR = (0, 0, 0)
CV2_TEXT_BG_COLOR = (255, 255, 255)
CV2_GEOMETRY_OUTLINE = (100, 100, 100)
CV2_ROI_CHECK_COLOR = (0, 0, 255)
CV2_ROI_POINT_COLOR = BGR_CYAN

CV2_POLY_COLORS = [
    (230, 216, 173),
    (144, 238, 144),
    (128, 128, 240),
    (181, 228, 255),
    (211, 211, 211),
    (203, 192, 255),
]

# ==========================================
# UNIFIED ZONE & ROI STYLES
# ==========================================
ZONE_STYLES = {
    "top_open": {"mpl": "#ff4500", "cv2": (0, 69, 255)},  # orangered
    "right_closed": {"mpl": "#00ff7f", "cv2": (127, 255, 0)},  # springgreen
    "bottom_open": {"mpl": "#1e90ff", "cv2": (255, 144, 30)},  # dodgerblue
    "left_closed": {"mpl": "#ba55d3", "cv2": (211, 85, 186)},  # mediumorchid
    "center": {"mpl": "#800080", "cv2": (128, 0, 128)},
    "none": {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "missing": {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "outlier": {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "zone_r1_c1": {"mpl": "#800080", "cv2": (128, 0, 128)},
}

# Fallback for dynamic/unknown zones
FALLBACK_ZONE_STYLE = {"mpl": "#00FFFF", "cv2": (255, 255, 0)}
