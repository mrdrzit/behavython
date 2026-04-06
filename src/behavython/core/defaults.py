# ==========================================
# APPLICATION & GUI SETTINGS
# ==========================================
APP_NAME = "Behavython"
MAIN_WINDOW_TITLE = "Behavython"
LOGO_WIDGET_NAME = "behavython_logo"

DEFAULT_STYLE = """
QWidget {
    background-color: #4B4B4B;
    color: #FFFFFF;
}
""".strip()

DEBUG_STYLE = """
/* 1. Global Background and Text */
QWidget {
    background-color: #1E1E1E;
    color: #CCCCCC;
}

/* 2. Text Areas and Logs (Make it look like a console) */
QTextEdit, QPlainTextEdit, QLineEdit {
    background-color: #121212;
    color: #CE9178; /* VS Code String Orange */
    border: 1px solid #d97706; /* Amber warning border */
    border-radius: 4px;
    font-family: "Consolas", monospace;
    padding: 4px;
}

/* 3. Buttons: Warning Amber outine, fills on hover */
QPushButton {
    background-color: #2D2D2D;
    color: #d97706;
    border: 1px solid #d97706;
    border-radius: 6px;
    font-weight: bold;
    padding: 6px;
}

QPushButton:hover {
    background-color: #d97706;
    color: #1E1E1E;
}

QPushButton:pressed {
    background-color: #b45309;
    border: 1px solid #b45309;
    color: #FFFFFF;
}

/* 4. Progress Bar: Shift from blue to amber */
QProgressBar {
    border: 1px solid #3C3C3C;
    border-radius: 4px;
    background-color: #252526;
    text-align: center;
    color: #FFFFFF;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: #d97706;
    border-radius: 2px;
}

/* 5. ComboBoxes (Dropdowns) */
QComboBox {
    background-color: #252526;
    border: 1px solid #3C3C3C;
    border-radius: 4px;
    padding-left: 8px;
    color: #D4D4D4;
}

QComboBox::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #1E1E1E;
    color: #CE9178;
    selection-background-color: #d97706;
    selection-color: #1E1E1E;
}

/* 6. Checkboxes */
QCheckBox {
    color: #D4D4D4;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    background-color: #252526;
    border: 1px solid #555555;
    border-radius: 4px;
}

QCheckBox::indicator:checked {
    background-color: #d97706;
    border: 1px solid #d97706;
}
""".strip()

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
    "image": (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    "position": ("filtered.csv",),
    "skeleton": ("filtered_skeleton.csv",),
    "roi": ("_roi.csv",),
}

VALID_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

# ==========================================
# PLOTTING & VISUALIZATION STYLES
# ==========================================
PLOT_COLORMAP = "inferno"
PLOT_TRAJECTORY_COLOR = "orangered"
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

ZONE_FALLBACK_COLORS = [
    "cyan", "lime", "orange", "pink", "teal", 
    "coral", "gold", "white", "brown"
]

PLOT_GEOMETRY_COLORS = [
    "red", "blue", "green", "yellow", "purple", 
    "cyan", "magenta", "orange", "lime"
]

ANIMATION_POLY_COLORS = [
    "lightblue", "lightgreen", "lightcoral", 
    "moccasin", "lightgrey", "pink"
]

# ==========================================
# OPENCV VIDEO ANIMATION STYLES (BGR Format)
# ==========================================
CV2_TRAJECTORY_COLOR = (255, 0, 0)
CV2_CROSSING_COLOR = (0, 0, 255)
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

# ------------------------------------------------------------------
# CORE COLOR PALETTE 
# ------------------------------------------------------------------
# We use HEX for Matplotlib and BGR Tuples for OpenCV
HEX_ORANGERED = "#FF4500"
BGR_ORANGERED = (0, 69, 255)  # Matches Orangered

HEX_CYAN = "#00FFFF"
BGR_CYAN = (255, 255, 0)

HEX_MAGENTA = "#FF00FF"
BGR_MAGENTA = (255, 0, 255)

# ------------------------------------------------------------------
# UNIFIED ZONE & ROI STYLES
# ------------------------------------------------------------------
# Maps zone names to their visual representation in both Plotting and Video
ZONE_STYLES = {
    "top_open":     {"mpl": "#FF0000", "cv2": (0, 0, 255)},     # Red
    "right_closed": {"mpl": "#0000FF", "cv2": (255, 0, 0)},     # Blue
    "bottom_open":  {"mpl": "#008000", "cv2": (0, 128, 0)},     # Green
    "left_closed":  {"mpl": "#FFFF00", "cv2": (0, 255, 255)},   # Yellow
    "center":       {"mpl": "#800080", "cv2": (128, 0, 128)},   # Purple
    "none":         {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "missing":      {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    "outlier":      {"mpl": HEX_MAGENTA, "cv2": BGR_MAGENTA},
    # Open Field Grid Defaults
    "zone_r1_c1":   {"mpl": "#800080", "cv2": (128, 0, 128)},   # Center (Purple)
}

# Fallback for dynamic/unknown zones
FALLBACK_ZONE_STYLE = {"mpl": "#00FFFF", "cv2": (255, 255, 0)} # Cyan

# ------------------------------------------------------------------
# REFINED PLOTTING CONSTANTS
# ------------------------------------------------------------------
PLOT_TRAJECTORY_COLOR = HEX_ORANGERED
CV2_TRAJECTORY_COLOR = BGR_ORANGERED
CV2_CROSSING_COLOR = (255, 255, 255) # White for high contrast on events