import os
import json
from typing import Callable, Any

from behavython.pipeline.models import Animal
from behavython.pipeline.metrics import preprocess_animal, compute_roi_interaction
from behavython.services.validation import validate_analysis_request
from behavython.core.utils import group_analysis_files

def analyze_animal(animal: Animal, request: Any) -> dict:
    """
    Main analysis entry point for a single experimental unit.

    Pipeline:
        1. Preprocess raw animal data → normalized numpy-based structure
        2. Compute interaction data (ROI, collisions) [optional]
        3. Compute movement metrics
        4. Compute exploration metrics
        5. Aggregate outputs

    Returns:
        dict: flat dictionary with all computed metrics
    """

    # 1. Preprocessing
    data = preprocess_animal(animal, request)

    # 2. ROI / interaction
    interaction_data = compute_roi_interaction(data)  # Passing request if needed by your implementation

    # 3. Movement metrics (core, always present)
    # movement_metrics = compute_movement_metrics(data, request)
    movement_metrics = {}

    # 4. Exploration metrics
    exploration_metrics = {}
    if interaction_data is not None:
        pass
        # exploration_metrics = compute_exploration_metrics(interaction_data, data, request)

    # 5. Spatial metrics
    spatial_metrics = {}
    # spatial_metrics = compute_spatial_metrics(data, request)

    # 6. Aggregate results
    results = {
        "animal_id": animal.id,
        **movement_metrics,
        **exploration_metrics,
        **spatial_metrics,
    }

    return results


def run_analysis_workflow(request: Any, progress: Callable = None, log: Callable = None, warning: Callable = None) -> dict:
    """
    Orchestrates the batch processing of multiple animals.
    """
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
            has_issues = True

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
        "results": results,  # Assuming you want to return the actual data
    }
