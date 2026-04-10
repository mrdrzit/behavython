import os
import yaml
from flask import json
from typing import Any
from pathlib import Path
from behavython.core.defaults import VALID_VIDEO_EXTENSIONS
from behavython.core.paths import USER_BIN_ROOT, USER_MODELS_ROOT
from behavython.pipeline.models import AnalysisRequest


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


def validate_yaml_text(yaml_text: str) -> dict[str, Any]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError("YAML root is not a dictionary.")
    return data


def validate_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        return validate_yaml_text(handle.read())


def validate_json_config(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data is not None and isinstance(data, dict)
    except Exception:
        return None


def validate_analysis_request(request: AnalysisRequest) -> list[str]:
    errors: list[str] = []

    if not request.input_files:
        errors.append("No analysis input files were selected.")

    if not request.output_folder:
        errors.append("No output folder was selected.")

    for path in request.input_files:
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")

    return errors


def is_ffmpeg_installed() -> bool:
    ffmpeg_exe = USER_BIN_ROOT / "ffmpeg.exe"
    ffprobe_exe = USER_BIN_ROOT / "ffprobe.exe"
    return ffmpeg_exe.exists() and ffprobe_exe.exists()


def is_model_installed(model_name: str) -> bool:
    model_dir = USER_MODELS_ROOT / model_name
    return model_dir.exists() and any(model_dir.iterdir())
