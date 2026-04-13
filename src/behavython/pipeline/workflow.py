import os
import math
import json
import logging
from tqdm import tqdm
from typing import Callable
from behavython.core.defaults import MAZE_EXPERIMENT_TYPES, MIN_EVENT_FRAMES, LINK_WINDOW
from behavython.pipeline import filters, geometry  # noqa: F401
from behavython.pipeline.models import AnalysisRequest, Animal, MazeAnimal
from behavython.pipeline.metrics import (
    compute_crossings,
    compute_social_behavior_metrics,
    compute_zone_metrics,
    extract_collision_coordinates,
    preprocess_animal,
    compute_roi_interaction,
    compute_movement_metrics,
    compute_spatial_metrics,
    compute_exploration_metrics,
)
from behavython.services.validation import validate_analysis_request
from behavython.pipeline.export import export_results_to_parquet, export_summary_metrics
from behavython.pipeline.plotting import plot_animal_analysis
from behavython.core.utils import group_analysis_files

console_logger = logging.getLogger("behavython.console")


def analyze_animal(animal: Animal, request: AnalysisRequest) -> dict:
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

    data = preprocess_animal(animal, request)
    interaction_data = compute_roi_interaction(data)
    collision_data = extract_collision_coordinates(interaction_data)

    start, end = data["analysis_range"]
    movement_metrics = compute_movement_metrics(
        centro_x=data["coords"]["center"]["x"][start:end],
        centro_y=data["coords"]["center"]["y"][start:end],
        fps=data["fps"],
        scale_x=data["scaling"]["x"],
        scale_y=data["scaling"]["y"],
        threshold=data["threshold"],
    )

    exploration_metrics = compute_exploration_metrics(interaction_data, data["fps"])
    spatial_metrics = compute_spatial_metrics(data, movement_metrics)

    latent_social_behavior_metrics = compute_social_behavior_metrics(
        collisions_df=interaction_data["collisions"], fps=data["fps"], min_frames=MIN_EVENT_FRAMES, link_window=LINK_WINDOW
    )

    return {
        "animal_id": animal.id,
        "experiment_type": request.options.experiment_type,
        **movement_metrics,
        **exploration_metrics,
        **spatial_metrics,
        **collision_data,
        **latent_social_behavior_metrics,
        "collisions_df": interaction_data["collisions"],
    }


def analyze_maze_animal(maze_animal: MazeAnimal, request: AnalysisRequest) -> dict:
    """
    The new pipeline specifically for Open Field and Plus Maze.
    Strings together pure extraction, filtering, geometry, and spatial events.
    """

    # 1. Calculate Pixel-to-CM Scaling Factors
    if maze_animal.experiment_type == "open_field":
        tl, tr, _, bl = maze_animal.arena_corners
        # Euclidean distance between top-left and top-right
        pixel_width = math.dist(tl, tr)
        # Euclidean distance between top-left and bottom-left
        pixel_height = math.dist(tl, bl)

    elif maze_animal.experiment_type == "plus_maze":
        pts = maze_animal.maze_points
        # Mapping the furthest arm extremities
        pixel_width = math.dist(pts[11], pts[4])  # Closed arms distance
        pixel_height = math.dist(pts[2], pts[7])  # Open arms distance

    # Scale = Physical Size (cm) / Pixel Size (px)
    scale_x = request.options.arena_width / pixel_width if pixel_width > 0 else 1.0
    scale_y = request.options.arena_height / pixel_height if pixel_height > 0 else 1.0

    # 2. Data extraction
    raw_x, raw_y = maze_animal.get_primary_tracking_data(preferred_bodypart="center")
    fps = request.options.frames_per_second
    max_time = request.options.task_duration
    trim_frames = int(request.options.trim_amount * fps)
    total_frames = int(max_time * fps)
    start = trim_frames
    end = min(trim_frames + total_frames, len(raw_x))

    sliced_x = raw_x.iloc[start:end].reset_index(drop=True)
    sliced_y = raw_y.iloc[start:end].reset_index(drop=True)

    # 3. Filtering (disabled for now)
    # filtered_x, filtered_y = filters.apply_rolling_window_filter(sliced_x, sliced_y, window_size=5, jump_threshold=150)
    filtered_x, filtered_y = sliced_x, sliced_y

    # 4. Kinematics (reusing your existing compute_movement_metrics)
    # We pass scale_x/y as 1.0 here if pixel-to-cm conversion is handled in preprocessing
    kinematics = compute_movement_metrics(
        centro_x=filtered_x.values, centro_y=filtered_y.values, fps=fps, scale_x=scale_x, scale_y=scale_y, threshold=request.options.threshold
    )

    # 5. Geometry & Spatial Events
    if maze_animal.experiment_type == "open_field":
        arena_polygons = geometry.build_grid_open_field_geometry(maze_animal.arena_corners)
    else:
        arena_polygons = geometry.build_plus_maze_geometry(maze_animal.maze_points)

    # Pure extraction and legacy translation
    zone_metrics = compute_zone_metrics(filtered_x, filtered_y, arena_polygons, fps)
    crossings = compute_crossings(zone_metrics["spatial_state_array"], maze_animal.experiment_type)

    return {
        "animal_id": maze_animal.animal.id,
        "experiment_type": maze_animal.experiment_type,
        "filtered_x": filtered_x,
        "filtered_y": filtered_y,
        "spatial_state_array": zone_metrics["spatial_state_array"],
        "time_in_zones_s": zone_metrics["time_in_zones_s"],
        "crossings": crossings,
        **kinematics,
    }


def run_analysis_workflow(request: AnalysisRequest, progress: Callable = None, log: Callable = None, warning: Callable = None) -> dict:
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
            image_path=files["image"][0] if files["image"] else None,
            skeleton_csv=files["skeleton"][0] if files["skeleton"] else None,
            roi_csv=files["roi"][0] if files["roi"] else None,
            video_path=files["video"][0] if files["video"] else None,
            experiment_type=request.options.experiment_type,
        )
        animals.append(animal)

    valid_animals_for_processing = [a for a in animals if a.eligible]
    initial_invalid_count = len(animals) - len(valid_animals_for_processing)

    if log:
        log.emit("resume", f"Valid animals for processing: {len(valid_animals_for_processing)} | Initial invalid: {initial_invalid_count}")

    results = []
    has_issues = False

    for index, animal in tqdm(enumerate(valid_animals_for_processing, start=1), total=len(valid_animals_for_processing), position=0):
        try:
            if request.options.experiment_type in MAZE_EXPERIMENT_TYPES:
                arena_corners = []
                maze_points = []

                if request.config_path and os.path.exists(request.config_path):
                    with open(request.config_path, "r", encoding="utf-8") as f:
                        config_data = json.load(f)
                        if request.options.experiment_type == "open_field":
                            arena_corners = config_data.get("arena_corners", [])
                        elif request.options.experiment_type == "plus_maze":
                            maze_points = config_data.get("maze_points", [])
                else:
                    raise FileNotFoundError("Arena configuration file is missing or invalid.")

                maze_dto = MazeAnimal(
                    animal=animal, experiment_type=request.options.experiment_type, arena_corners=arena_corners, maze_points=maze_points
                )
                result = analyze_maze_animal(maze_dto, request)
            else:
                result = analyze_animal(animal, request)

            results.append(result)

            if request.options.plot_options == "plotting_enabled":
                plot_animal_analysis(animal, result, request)

        except Exception as e:
            animal.logs.append({"level": "error", "message": str(e), "context": "analysis"})
            animal.eligible = False
            has_issues = True

        if progress:
            progress.emit(round((index / len(valid_animals_for_processing)) * 100))

    if results:
        if log:
            log.emit("resume", "Exporting results to Parquet...")
        try:
            export_results_to_parquet(results, request.output_folder)
            export_summary_metrics(results, request.output_folder, log=log)
        except Exception as e:
            if warning:
                warning.emit("Export Error", f"Failed to save Parquet files: {str(e)}")
            has_issues = True

    all_logs = []
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

    log_path = None
    if has_issues and all_logs:
        log_path = os.path.join(request.output_folder, "analysis_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(all_logs, f, indent=2)

        if log:
            log.emit("resume", f"Log file written to: {log_path}")
        if warning:
            warning.emit("Warning", f"Analysis completed with issues. Log saved to:\n{log_path}")

    final_valid_count = sum(1 for a in animals if a.eligible)
    final_invalid_count = sum(1 for a in animals if not a.eligible)

    return {
        "kind": "analysis",
        "output_path": request.output_folder,
        "rows": len(animals),
        "valid_animals": final_valid_count,
        "invalid_animals": final_invalid_count,
        "log_path": log_path,
        "results": results,
    }
