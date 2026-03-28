from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from behavython.services.yaml_repair_service import load_or_repair_dlc_yaml


def load_deeplabcut():
    if "deeplabcut" not in sys.modules:
        import deeplabcut  # type: ignore
    else:
        deeplabcut = sys.modules["deeplabcut"]
    return deeplabcut


def prepare_dlc_config(config_path: str) -> tuple[dict[str, Any], str, bool]:
    """
    Returns:
        config_dict,
        usable_config_path,
        was_repaired
    """
    config_dict, resolved_path, was_repaired = load_or_repair_dlc_yaml(config_path)
    return config_dict, str(Path(resolved_path)), was_repaired
