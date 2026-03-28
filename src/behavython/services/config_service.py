from __future__ import annotations

import json


def load_json_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_json_config(path: str) -> bool:
    try:
        load_json_config(path)
        return True
    except Exception:
        return False
