from __future__ import annotations

import os

from behavython.config.defaults import VALID_VIDEO_EXTENSIONS


def validate_config_path(path: str) -> list[str]:
    errors: list[str] = []

    if not path:
        errors.append("No config.yaml was selected.")
    elif not os.path.exists(path):
        errors.append(f"Config path does not exist: {path}")
    elif not path.lower().endswith(".yaml"):
        errors.append("Config path must be a .yaml file.")

    return errors


def validate_video_paths(video_paths: list[str]) -> list[str]:
    errors: list[str] = []

    if not video_paths:
        errors.append("No videos were selected.")

    for path in video_paths:
        if not os.path.exists(path):
            errors.append(f"Missing video: {path}")
        elif not path.lower().endswith(VALID_VIDEO_EXTENSIONS):
            errors.append(f"Invalid video extension: {path}")

    extensions = {os.path.splitext(path)[1].lower() for path in video_paths}
    if len(extensions) > 1:
        errors.append("All videos must have the same extension.")

    return errors