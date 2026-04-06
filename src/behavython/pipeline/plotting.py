import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from behavython.pipeline.models import Animal, AnalysisRequest
from behavython.pipeline import geometry


def plot_animal_analysis(animal: Animal, result: dict, request: AnalysisRequest) -> None:
    """
    Main plotting router. Directs the result dictionary to the correct
    visualization generator based on the experiment type.
    """
    if request.options.experiment_type in ["open_field", "plus_maze"]:
        _plot_maze_analysis(animal, result, request)
    else:
        _plot_object_recognition(animal, result, request)


def _plot_object_recognition(animal: Animal, result: dict, request: AnalysisRequest) -> None:
    """
    Plotting logic for Object Recognition / Social Exploration tasks.
    """
    save_folder = request.output_folder
    animal_name = animal.id
    animal_image = animal.image

    x_pos = result["raw_x_pos_array"]
    y_pos = result["raw_y_pos_array"]
    x_collisions = result["x_collision_data"]
    y_collisions = result["y_collision_data"]
    position_grid = result["position_grid"]
    accumulate_distance = result["accumulate_distance_array"]

    fps = request.options.frames_per_second
    time_vector = np.arange(0, len(x_pos) / fps, 1 / fps)[: len(x_pos)]

    max_w, max_h = map(int, request.options.max_fig_res)
    img_w, img_h = animal.exp_dimensions()

    ratio = min(max_w / img_w, max_h / img_h) if img_w > 0 and img_h > 0 else 1
    new_size = (int(img_w * ratio / 100), int(img_h * ratio / 100))

    plt.ioff()
    try:
        # 1. Heatmap
        fig1, ax1 = plt.subplots(figsize=new_size)
        ax1.set_title(f"Center Position Heatmap: {animal_name}")
        ax1.imshow(position_grid, cmap="inferno", interpolation="bessel")
        ax1.axis("off")
        fig1.savefig(os.path.join(save_folder, f"{animal_name} - Overall heatmap of the mice's position (nose).png"), bbox_inches="tight")

        # 2. Exploration Map
        fig2, ax2 = plt.subplots(figsize=new_size)
        ax2.set_title(f"Exploration Map: {animal_name}")
        if len(x_collisions) > 1:
            sns.kdeplot(x=x_collisions, y=y_collisions, ax=ax2, cmap="inferno", fill=True, alpha=0.5)
        if animal_image is not None:
            ax2.imshow(animal_image, interpolation="bessel")
        ax2.axis("off")
        fig2.savefig(os.path.join(save_folder, f"{animal_name} - Overall exploration by ROI.png"), bbox_inches="tight")

        # 3. Accumulated Distance
        fig3, ax3 = plt.subplots(figsize=new_size)
        ax3.set_title(f"Accumulated Distance: {animal_name}")
        ax3.plot(time_vector, accumulate_distance)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Distance (cm)")
        ax3.grid(True)
        fig3.savefig(os.path.join(save_folder, f"{animal_name} - Distance accumulated over time.png"), bbox_inches="tight")

        # 4. Trajectory
        fig4, ax4 = plt.subplots(figsize=new_size)
        ax4.set_title(f"Movement Trajectory: {animal_name}")
        ax4.plot(x_pos, y_pos, color="orangered", linewidth=1.5)
        if animal_image is not None:
            ax4.imshow(animal_image, interpolation="bessel", alpha=0.8)
        ax4.axis("off")
        fig4.savefig(os.path.join(save_folder, f"{animal_name} - Animal movement in the arena (nose).png"), bbox_inches="tight")

    finally:
        plt.close("all")


def _plot_maze_analysis(animal: Animal, result: dict, request: AnalysisRequest) -> None:
    """
    New plotting logic specifically for Open Field and Plus Maze.
    Recreates the legacy geometric overlays and trajectory plots.
    """
    save_folder = request.output_folder
    animal_name = animal.id
    animal_image = animal.image

    # Extract maze-specific metrics
    x_pos = result.get("filtered_x", [])
    y_pos = result.get("filtered_y", [])
    accumulate_distance = result.get("accumulate_distance_array", [])

    fps = request.options.frames_per_second
    time_vector = np.arange(0, len(x_pos) / fps, 1 / fps)[: len(x_pos)]

    max_w, max_h = map(int, request.options.max_fig_res)
    img_w, img_h = animal.exp_dimensions()

    ratio = min(max_w / img_w, max_h / img_h) if img_w > 0 and img_h > 0 else 1
    new_size = (int(img_w * ratio / 100), int(img_h * ratio / 100))

    plt.ioff()
    try:
        # 1. Trajectory Plot
        fig1, ax1 = plt.subplots(figsize=new_size)
        ax1.set_title(f"Movement Trajectory: {animal_name}")
        ax1.plot(x_pos, y_pos, color="orangered", linewidth=1.5)
        if animal_image is not None:
            ax1.imshow(animal_image, interpolation="bessel", alpha=0.8)
        ax1.axis("off")
        fig1.savefig(os.path.join(save_folder, f"{animal_name} - Movement in the arena.png"), bbox_inches="tight")

        # 2. Accumulated Distance Plot
        if len(accumulate_distance) > 0:
            fig2, ax2 = plt.subplots(figsize=new_size)
            ax2.set_title(f"Accumulated Distance: {animal_name}")
            ax2.plot(time_vector, accumulate_distance)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Distance (cm)")
            ax2.grid(True)
            fig2.savefig(os.path.join(save_folder, f"{animal_name} - Distance accumulated over time.png"), bbox_inches="tight")

        # 3. Geometry Overlay (Legacy check_maze_geometry)
        if request.config_path and os.path.exists(request.config_path):
            with open(request.config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)

            if request.options.experiment_type == "open_field":
                polygons = geometry.build_grid_open_field_geometry(config_data.get("arena_corners", []))
            elif request.options.experiment_type == "plus_maze":
                polygons = geometry.build_plus_maze_geometry(config_data.get("maze_points", []))
            else:
                polygons = {}

            if polygons:
                fig3, ax3 = plt.subplots(figsize=new_size)
                ax3.set_title(f"Maze Geometry Overlay: {animal_name}")
                if animal_image is not None:
                    ax3.imshow(animal_image)

                # Standard color-coding for maze sections
                colors = ["red", "blue", "green", "yellow", "purple", "cyan", "magenta", "orange", "lime"]

                for i, (zone_name, poly) in enumerate(polygons.items()):
                    x, y = poly.exterior.xy
                    color = colors[i % len(colors)]
                    ax3.plot(x, y, color=color, linewidth=2, label=zone_name)

                ax3.axis("off")
                fig3.savefig(os.path.join(save_folder, f"{animal_name} - Maze Geometry.png"), bbox_inches="tight")
        _plot_entries(animal, result, request)

    finally:
        plt.close("all")


def _plot_entries(animal: Animal, result: dict, request: AnalysisRequest) -> None:
    """
    Plots the color-coded entries for the animal on a background image.
    Works universally for both Open Field and Plus Maze by parsing the spatial_state_array.
    """
    save_folder = request.output_folder
    animal_name = animal.id
    animal_image = animal.image

    # Extract the vectorized coordinates and states
    x_pos = result.get("filtered_x")
    y_pos = result.get("filtered_y")

    # We must retrieve the spatial state array directly from the result
    # We need to make sure workflow.py adds "spatial_state_array" to the result dict!
    states = result.get("spatial_state_array")

    if x_pos is None or states is None or len(x_pos) == 0:
        return

    # Calculate new image dimensions
    max_w, max_h = map(int, request.options.max_fig_res)
    img_w, img_h = animal.exp_dimensions()
    ratio = min(max_w / img_w, max_h / img_h) if img_w > 0 and img_h > 0 else 1
    new_size = (int(img_w * ratio / 100), int(img_h * ratio / 100))
    dpi = 150

    plt.ioff()
    fig, ax = plt.subplots(figsize=new_size, dpi=dpi)
    fig.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
    ax.set_title(f"Entries for animal {animal_name}", fontsize=10)

    if animal_image is not None:
        ax.imshow(animal_image)
    ax.axis("off")

    # Standardize the color mapping to match your legacy plots
    color_map = {
        "top_open": "red",
        "right_closed": "blue",
        "bottom_open": "green",
        "left_closed": "yellow",
        "center": "purple",
        "none": "magenta",
        "missing": "magenta",
        "outlier": "magenta",
    }

    # Fallback colors for the dynamically generated 3x3 Open Field grid zones
    fallback_colors = ["cyan", "lime", "orange", "pink", "teal", "coral", "gold", "white", "brown"]

    unique_zones = states.unique()

    # Plot each zone using Pandas boolean masking (much faster than loops)
    for i, zone in enumerate(unique_zones):
        mask = states == zone
        color = color_map.get(zone)

        if not color:
            color = fallback_colors[i % len(fallback_colors)]

        ax.scatter(x_pos[mask], y_pos[mask], s=6, color=color, label=zone)

    # Optional: Add legend if you want to see the color keys
    # ax.legend(loc="upper right", fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{animal_name} - Entries.png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
