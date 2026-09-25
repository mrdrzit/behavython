import os
import yaml
from flask import json
from typing import Any, Optional
from pathlib import Path
from importlib.util import find_spec
from behavython.core.defaults import ANALYSIS_REQUIRED_SUFFIXES
from behavython.core.paths import USER_MODELS_ROOT
from behavython.pipeline.models import AnalysisRequest
from behavython.core.exceptions import AnalysisError, UnsupportedBackendError, MissingBackendError


def validate_config_path(path: str) -> list[str]:
    errors: list[str] = []

    if not path:
        errors.append("No configuration file was selected.")
    elif not os.path.exists(path):
        errors.append(f"Config path does not exist: {path}")
    elif not (path.lower().endswith(".yaml") or path.lower().endswith(".yml")):
        errors.append("Config path must be a .yaml or .yml file.")

    return errors


def validate_video_paths(video_paths: list[str]) -> list[str]:
    errors: list[str] = []

    if not video_paths:
        errors.append("No videos were selected.")

    for path in video_paths:
        if not os.path.exists(path):
            errors.append(f"Missing video: {path}")
        elif not path.lower().endswith(ANALYSIS_REQUIRED_SUFFIXES["video"]):
            errors.append(f"Invalid video extension: {path}")

    extensions = {os.path.splitext(path)[1].lower() for path in video_paths}
    if len(extensions) > 1:
        errors.append("All videos must have the same extension.")

    return errors


def validate_yaml_text(yaml_text: str) -> dict[str, Any]:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise AnalysisError("YAML root is not a dictionary.")
    return data


def validate_yaml_file(path: str | Path) -> dict[str, Any]:
    yaml_path = Path(path)
    with yaml_path.open("r", encoding="utf-8") as handle:
        return validate_yaml_text(handle.read())


def validate_json_config(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data is not None and isinstance(data, dict):
            return data
        return None
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
    from behavython.core.utils import resolve_binary

    try:
        resolve_binary("ffmpeg")
        resolve_binary("ffprobe")
        return True
    except RuntimeError:
        return False


def is_model_installed(model_name: str) -> bool:
    model_dir = USER_MODELS_ROOT / model_name
    return model_dir.exists() and any(model_dir.iterdir())


ENGINE_ALIASES: dict[str, str] = {
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "torch": "pytorch",
}

BACKEND_PACKAGES: dict[str, list[str]] = {
    "pytorch": ["torch"],
    "tensorflow": ["tensorflow", "tf_slim"],
}


def _infer_engine_from_folders(project_root: Path) -> Optional[str]:
    """Infers engine based on DeepLabCut model folder conventions."""
    if not project_root.exists() or not project_root.is_dir():
        return None

    has_pytorch_models = (project_root / "dlc-models-pytorch").is_dir()
    has_tf_models = (project_root / "dlc-models").is_dir()

    if has_pytorch_models and not has_tf_models:
        return "pytorch"
    if has_tf_models and not has_pytorch_models:
        return "tensorflow"
    return None


def validate_dlc_backend(config: dict[str, Any], config_path: str | Path | None = None) -> str:
    """
    Validate that the backend required by the DLC network is installed.

    DeepLabCut networks without an explicit ``engine`` key check for
    engine-specific model directories, falling back to TensorFlow for
    pre-3.0 compatibility.

    Returns:
        The backend that will be used: ``"pytorch"`` or ``"tensorflow"``.

    Raises:
        UnsupportedBackendError: If the config specifies an unknown engine.
        MissingBackendError: If the required backend is not installed.
    """
    raw_engine = config.get("engine")
    engine_candidate: Optional[str] = None

    if isinstance(raw_engine, str) and raw_engine.strip():
        normalized = raw_engine.strip().lower()
        if normalized not in ENGINE_ALIASES:
            supported = ", ".join(sorted(set(ENGINE_ALIASES.values())))
            raise UnsupportedBackendError(f"Unsupported DeepLabCut engine: {raw_engine!r}. Expected one of: {supported}.")
        engine_candidate = ENGINE_ALIASES[normalized]

    if engine_candidate is None:
        project_dir: Optional[Path] = None
        if config_path:
            project_dir = Path(config_path).resolve().parent
        elif config.get("project_path"):
            project_dir = Path(config["project_path"]).resolve()

        if project_dir:
            engine_candidate = _infer_engine_from_folders(project_dir)

        if engine_candidate is None:
            engine_candidate = "tensorflow"

    required_packages = BACKEND_PACKAGES.get(engine_candidate, [])
    missing_packages = [pkg for pkg in required_packages if find_spec(pkg) is None]

    if missing_packages:
        pkgs_str = ", ".join(repr(p) for p in missing_packages)
        raise MissingBackendError(f"This DeepLabCut network requires the {engine_candidate} backend, but {pkgs_str} is not installed.")

    return engine_candidate
