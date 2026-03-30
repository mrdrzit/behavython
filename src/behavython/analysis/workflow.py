from __future__ import annotations

import os
import json
from behavython.analysis.animal import Animal
from behavython.analysis.models import analysis_request
from behavython.analysis.analysis import analyze_animal
from behavython.analysis.validation import validate_analysis_request
from behavython.shared.input_resolver import group_analysis_files


def run_analysis_workflow(request: analysis_request, progress=None, log=None, warning=None) -> dict:
    errors = validate_analysis_request(request)
    if errors:
        raise ValueError("\n".join(errors))

    if log:
        log.emit("resume", "Starting analysis workflow...")

    groups = group_analysis_files(request.input_files)

    if log:
        log.emit("resume", f"Detected {len(groups)} animal groups")

    animals = []
    for index, group in enumerate(groups, start=1):
        files = group["files"]

        animal = Animal(
            animal_id=group["animal_id"],
            position_csv=files["position"][0] if files["position"] else None,
            skeleton_csv=files["skeleton"][0] if files["skeleton"] else None,
            roi_csv=files["roi"][0] if files["roi"] else None,
            image_path=files["image"][0] if files["image"] else None,
            video_path=files["video"][0] if files["video"] else None,
        )
        animals.append(animal)

    valid_animals = [a for a in animals if a.eligible]
    invalid_animals = [a for a in animals if not a.eligible]
    rows = []

    for _, animal in enumerate(animals, start=1):
        rows.append(
            {
                "animal_id": animal.id,
                "eligible": animal.eligible,
                "missing_files": ", ".join(animal.missing_files) if not animal.eligible else "",
                "has_video": animal.video_path is not None,
            }
        )

    if log:
        log.emit("resume", f"Valid animals: {len(valid_animals)} | Invalid animals: {len(invalid_animals)}")

    all_logs = []
    has_issues = False

    for animal in animals:
        if animal.logs:
            has_issues = True

        for entry in animal.logs:
            all_logs.append(
                {
                    "animal_id": animal.id,
                    "level": entry["level"],
                    "message": entry["message"],
                    "context": entry["context"],
                }
            )

    results = []
    for index, animal in enumerate(valid_animals, start=1):
        if log:
            log.emit("resume", f"Analyzing {animal.id}")

        try:
            result = analyze_animal(animal, request)
            results.append(result)

        except Exception as e:
            animal.logs.append({"level": "error", "message": str(e), "context": "analysis"})
            animal.eligible = False

        if progress:
            progress.emit(round((index / len(valid_animals)) * 100))

    log_path = None
    if has_issues:
        log_path = os.path.join(request.output_folder, "analysis_log.json")

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, indent=2)

        if log:
            log.emit("resume", f"Log file written to: {log_path}")

        if warning:
            warning.emit("Warning", f"Analysis completed with issues. Log saved to:\n{log_path}")

    return {
        "kind": "analysis",
        "output_path": request.output_folder,
        "rows": len(animals),
        "valid_animals": len(valid_animals),
        "invalid_animals": len(invalid_animals),
        "log_path": log_path,
    }
