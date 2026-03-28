from __future__ import annotations

import os

from behavython.analysis.models import analysis_request


def validate_analysis_request(request: analysis_request) -> list[str]:
    errors: list[str] = []

    if not request.input_files:
        errors.append("No analysis input files were selected.")

    if not request.output_folder:
        errors.append("No output folder was selected.")

    for path in request.input_files:
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")

    return errors