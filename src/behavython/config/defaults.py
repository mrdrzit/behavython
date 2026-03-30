DEFAULT_ARENA_WIDTH = 30
DEFAULT_ARENA_HEIGHT = 30
DEFAULT_FRAMES_PER_SECOND = 30
DEFAULT_TASK_DURATION_SECONDS = 300
DEFAULT_TRIM_AMOUNT_SECONDS = 0

DEFAULT_PLOT_ENABLED = True
DEFAULT_CROP_VIDEO = False

DEFAULT_EXPERIMENT_TYPE_INDEX = 0
DEFAULT_ALGO_TYPE_INDEX = 0
DEFAULT_ANIMAL_INDEX = 0
DEFAULT_FIG_MAX_SIZE_INDEX = 1

ANALYSIS_REQUIRED_SUFFIXES = {
    "image": (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
    "position": ("filtered.csv",),
    "skeleton": ("filtered_skeleton.csv",),
    "roi": ("_roi.csv",),
}

VALID_VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov")

TOTAL_SESSION_STORAGE_QUOTA = 5

CANONICAL_BODYPARTS = [
    "nose",
    "left_ear",
    "right_ear",
    "center",
    "tail",
]

CANONICAL_REQUIRED_BODYPARTS = [
    "nose",
    "left_ear",
    "right_ear",
    "center",
]

BODY_PART_MAPPING = {
    "nose": "focinho",
    "left_ear": "orelhae",
    "right_ear": "orelhad",
    "center": "centro",
}

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

DEFAULT_ANALYSIS_PARAMETERS = {
    "frames_per_second": 30,
    "arena_width": 30,
    "arena_height": 30,
    "task_duration": 300,
    "trim_amount": 0,
    "threshold": 5,
}