# ==========================================
# APPLICATION & GUI SETTINGS
# ==========================================
APP_NAME = "Behavython"
MAIN_WINDOW_TITLE = "Behavython"
LOGO_WIDGET_NAME = "behavython_logo"

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

DEFAULT_STYLE = """
QWidget {
    background-color: #343434;
    color: #FFFFFF;
}
""" + "\n" + SCROLLBAR_STYLE

DEBUG_STYLE = """
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
""" + "\n" + SCROLLBAR_STYLE

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
LINK_WINDOW = 30
ANGLE_CONE_ENTER = 45.0
ANGLE_CONE_EXIT = 60.0
ROI_DELTA_LOOKBACK_FRAMES = 3
APPROACH_THRESHOLD_BL = 0.1
LOCOMOTION_THRESHOLD_BL = 0.02

DEFAULT_ANALYSIS_PARAMETERS = {
    "frames_per_second": 30,
    "arena_width": 30,
    "arena_height": 30,
    "task_duration": 300,
    "trim_amount": 0,
    "threshold": 5,
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
TOTAL_SESSION_STORAGE_QUOTA = 5

ANALYSIS_REQUIRED_SUFFIXES = {
    "image": (".png", ".jpg", ".jpeg", ".tiff"),
    "video": (".mp4", ".avi", ".mov"),
    "position": ("filtered.csv",),
    "skeleton": ("filtered_skeleton.csv",),
    "social_recognition_rois": ("_roi.csv",),
    "social_discrimination_rois": ("roil.csv", "roir.csv"),
    "maze_rois": (".json",),
}

LOGGING_NAME_MAP = {
    "behavython": "CORE",
    "behavython.dlc": "DLC",
    "behavython.console": "SYSTEM",
    "behavython.external": "EXTERNAL",
}

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
    "top_open": {"mpl": "#FF0000", "cv2": (0, 0, 255)},
    "right_closed": {"mpl": "#0000FF", "cv2": (255, 0, 0)},
    "bottom_open": {"mpl": "#008000", "cv2": (0, 128, 0)},
    "left_closed": {"mpl": "#FFFF00", "cv2": (0, 255, 255)},
    "center": {"mpl": "#800080", "cv2": (128, 0, 128)},
    "none": {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "missing": {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "outlier": {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "zone_r1_c1": {"mpl": "#800080", "cv2": (128, 0, 128)},
}

# Fallback for dynamic/unknown zones
FALLBACK_ZONE_STYLE = {"mpl": "#00FFFF", "cv2": (255, 255, 0)}
