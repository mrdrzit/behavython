from __future__ import annotations

import os
import pandas as pd
from collections import defaultdict
from behavython.analysis.models import analysis_request
from behavython.analysis.validation import validate_analysis_request


def _group_files_by_animal(paths: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        name = os.path.basename(path)
        stem = os.path.splitext(name)[0]

        if "_roi" in stem:
            animal_key = stem.replace("_roi", "")
        elif "filtered_skeleton" in stem:
            animal_key = stem.split("DLC_")[0].rstrip("_")
        elif "filtered" in stem:
            animal_key = stem.split("DLC_")[0].rstrip("_")
        else:
            animal_key = stem

        grouped[animal_key].append(path)

    return dict(grouped)


def run_analysis_workflow(request: analysis_request, progress=None, log=None, warning=None) -> dict:
    errors = validate_analysis_request(request)
    if errors:
        raise ValueError("\n".join(errors))

    if log:
        log.emit("resume", "Starting analysis workflow...")

    grouped = _group_files_by_animal(request.input_files)

    rows: list[dict] = []
    total = max(len(grouped), 1)

    for index, (animal, files) in enumerate(grouped.items(), start=1):
        lower_files = [os.path.basename(f).lower() for f in files]

        has_image = any(name.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")) for name in lower_files)
        has_position = any(name.endswith("filtered.csv") and not name.endswith("filtered_skeleton.csv") for name in lower_files)
        has_skeleton = any(name.endswith("filtered_skeleton.csv") for name in lower_files)
        has_roi = any(name.endswith("_roi.csv") for name in lower_files)

        rows.append(
            {
                "animal": animal,
                "file_count": len(files),
                "has_image": has_image,
                "has_position": has_position,
                "has_skeleton": has_skeleton,
                "has_roi": has_roi,
                "ready_for_full_analysis": all([has_image, has_position, has_skeleton, has_roi]),
            }
        )

        if log:
            log.emit("resume", f"Processed input manifest for {animal}")

        if progress:
            progress.emit(round((index / total) * 100))

    df = pd.DataFrame(rows)
    output_path = os.path.join(request.output_folder, "analysis_input_summary.xlsx")
    df.to_excel(output_path, index=False)

    if log:
        log.emit("resume", f"Analysis summary written to: {output_path}")

    return {
        "kind": "analysis",
        "output_path": output_path,
        "rows": len(df),
    }
