import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from behavython.pipeline.models import Animal, AnalysisRequest


def plot_animal_analysis(animal: Animal, result: dict, request: AnalysisRequest) -> None:
    """
    Plots behavioral maps and metrics for a single Animal.
    Uses the request options for scaling and output pathing.
    """
    save_folder = request.output_folder
    animal_name = animal.id
    animal_image = animal.image

    # Extract data from the results dictionary
    # Ensure these keys match what analyze_animal returns
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

    ratio = min(max_w / img_w, max_h / img_h)
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
        ax4.imshow(animal_image, interpolation="bessel", alpha=0.8)
        ax4.axis("off")
        fig4.savefig(os.path.join(save_folder, f"{animal_name} - Animal movement in the arena (nose).png"), bbox_inches="tight")

    finally:
        plt.close("all")
