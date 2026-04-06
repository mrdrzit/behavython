import matplotlib

matplotlib.use("Agg")
import cv2
import os
import json
import warnings
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from behavython.core.paths import FFMPEG_BIN_PATH
from matplotlib.animation import FuncAnimation
from behavython.pipeline.models import Animal, AnalysisRequest
from behavython.pipeline import geometry

# Import our new style variables
from behavython.core.defaults import (
    FALLBACK_ZONE_STYLE,
    PLOT_COLORMAP,
    PLOT_TRAJECTORY_COLOR,
    PLOT_TRAJECTORY_LINEWIDTH,
    PLOT_SCATTER_SIZE,
    ANIMATION_POLY_COLORS,
    CV2_TRAJECTORY_COLOR,
    CV2_CROSSING_COLOR,
    CV2_TEXT_COLOR,
    CV2_TEXT_BG_COLOR,
    CV2_GEOMETRY_OUTLINE,
    ZONE_STYLES,
)

console_logger = logging.getLogger("behavython.console")


def _configure_bundled_ffmpeg():
    """
    Locates the bundled ffmpeg binary using the centralized paths configuration
    and explicitly sets it for Matplotlib and the runtime environment.
    """
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    ffmpeg_exe = FFMPEG_BIN_PATH / ffmpeg_name

    if ffmpeg_exe.exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg_exe)
        os.environ["PATH"] = str(FFMPEG_BIN_PATH) + os.pathsep + os.environ.get("PATH", "")


_configure_bundled_ffmpeg()


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
        ax1.imshow(position_grid, cmap=PLOT_COLORMAP, interpolation="bessel")
        ax1.axis("off")
        fig1.savefig(os.path.join(save_folder, f"{animal_name} - Overall heatmap of the mice's position (nose).png"), bbox_inches="tight")

        # 2. Exploration Map
        fig2, ax2 = plt.subplots(figsize=new_size)
        ax2.set_title(f"Exploration Map: {animal_name}")
        if len(x_collisions) > 1:
            sns.kdeplot(x=x_collisions, y=y_collisions, ax=ax2, cmap=PLOT_COLORMAP, fill=True, alpha=0.5)
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
        ax4.plot(x_pos, y_pos, color=PLOT_TRAJECTORY_COLOR, linewidth=PLOT_TRAJECTORY_LINEWIDTH)
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
        fig1, ax1 = plt.subplots(figsize=new_size, dpi=150)
        fig1.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
        ax1.set_title(f"Movement Trajectory: {animal_name}")
        ax1.plot(x_pos, y_pos, color=PLOT_TRAJECTORY_COLOR, linewidth=PLOT_TRAJECTORY_LINEWIDTH)
        if animal_image is not None:
            ax1.imshow(animal_image, interpolation="bessel", alpha=0.8)
        ax1.axis("off")
        fig1.savefig(os.path.join(save_folder, f"{animal_name} - Movement in the arena.png"), bbox_inches="tight")

        # 2. Accumulated Distance Plot
        if len(accumulate_distance) > 0:
            fig2, ax2 = plt.subplots(figsize=new_size, dpi=150)
            fig2.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
            ax2.set_title(f"Accumulated Distance: {animal_name}")
            ax2.plot(time_vector, accumulate_distance)
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Distance (cm)")
            ax2.grid(True)
            fig2.savefig(os.path.join(save_folder, f"{animal_name} - Distance accumulated over time.png"), bbox_inches="tight")

        # 3. Geometry Overlay
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
                fig3, ax3 = plt.subplots(figsize=new_size, dpi=150)
                fig3.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
                ax3.set_title(f"Maze Geometry Overlay: {animal_name}")
                if animal_image is not None:
                    ax3.imshow(animal_image)

                sorted_zones = sorted(polygons.items(), key=lambda item: item[0] in ["center", "zone_r1_c1"])

                for i, (zone_name, poly) in enumerate(sorted_zones):
                    x, y = poly.exterior.xy
                    style = ZONE_STYLES.get(zone_name, FALLBACK_ZONE_STYLE)
                    z = 10 if zone_name in ["center", "zone_r1_c1"] else 2
                    ax3.plot(x, y, color=style["mpl"], linewidth=2, label=zone_name, zorder=z)

                ax3.axis("off")
                fig3.savefig(os.path.join(save_folder, f"{animal_name} - Maze Geometry.png"), bbox_inches="tight", dpi=150)

                if getattr(request.options, "generate_video", False):
                    _opencv_animate_maze_crossings(animal, result, request, polygons)
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

    x_pos = result.get("filtered_x")
    y_pos = result.get("filtered_y")
    states = result.get("spatial_state_array")

    if x_pos is None or states is None or len(x_pos) == 0:
        return

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

    unique_zones = states.unique()

    # Plot each zone using Pandas boolean masking
    for i, zone in enumerate(unique_zones):
        mask = states == zone
        style = ZONE_STYLES.get(zone, FALLBACK_ZONE_STYLE)
        ax.scatter(x_pos[mask], y_pos[mask], s=PLOT_SCATTER_SIZE, color=style["mpl"], label=zone)

    fig.tight_layout()
    fig.savefig(os.path.join(save_folder, f"{animal_name} - Entries.png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def _opencv_animate_maze_crossings(animal: Animal, result: dict, request: AnalysisRequest, polygons: dict) -> None:
    """
    Generates an MP4 validation video using OpenCV for massive performance gains.
    Directly manipulates pixel arrays instead of rendering Matplotlib figures.
    """
    console_logger.info(f"\nStarting OpenCV video generation for {animal.id}...")
    save_folder = request.output_folder
    animal_name = animal.id

    x_pos = result.get("filtered_x").values
    y_pos = result.get("filtered_y").values
    states = result.get("spatial_state_array").values
    fps = request.options.frames_per_second

    if len(x_pos) == 0 or len(states) == 0:
        return

    # 1. Identify all crossing events
    crossings = []
    previous_state = states[0]
    for i in range(1, len(states)):
        current_state = states[i]
        if current_state != previous_state:
            crossings.append({"index": i, "coordinates": (x_pos[i], y_pos[i]), "from": previous_state, "to": current_state})
            previous_state = current_state

    # 2. Setup Image Scaling & Canvas
    max_w, max_h = map(int, request.options.max_fig_res)
    img_w, img_h = animal.exp_dimensions()
    ratio = min(max_w / img_w, max_h / img_h) if img_w > 0 and img_h > 0 else 1
    new_w, new_h = int(img_w * ratio), int(img_h * ratio)

    # Create the base background
    if animal.image is not None:
        base_canvas = cv2.cvtColor(animal.image, cv2.COLOR_RGB2BGR)
        base_canvas = cv2.resize(base_canvas, (new_w, new_h))
    else:
        base_canvas = np.full((new_h, new_w, 3), 255, dtype=np.uint8)

    # 3. Draw the Maze Geometry (Semi-transparent overlay)
    overlay = base_canvas.copy()
    for zone_name, poly in polygons.items():
        px, py = poly.exterior.xy
        pts = np.array([[int(x * ratio), int(y * ratio)] for x, y in zip(px, py)], np.int32).reshape((-1, 1, 2))

        style = ZONE_STYLES.get(zone_name, FALLBACK_ZONE_STYLE)
        cv2.fillPoly(overlay, [pts], style["cv2"])  # Correct BGR color
        cv2.polylines(overlay, [pts], isClosed=True, color=CV2_GEOMETRY_OUTLINE, thickness=1)

    # Apply transparency (alpha blending)
    cv2.addWeighted(overlay, 0.3, base_canvas, 0.7, 0, base_canvas)

    # 4. Setup Video Writer
    save_path = os.path.join(save_folder, f"{animal_name} - Validation Video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (new_w, new_h))

    # Pre-scale all tracking coordinates to match the resized image
    scaled_x = (x_pos * ratio).astype(np.int32)
    scaled_y = (y_pos * ratio).astype(np.int32)

    active_annotations = []
    frames_to_keep_text = int(fps * 2)

    # 5. Render Loop
    for frame_idx in tqdm(range(len(scaled_x)), desc=f"Rendering {animal_name}", unit="frame", leave=False):
        frame_img = base_canvas.copy()

        # Draw Trajectory Trail (last 15 frames)
        trail_start = max(0, frame_idx - 15)
        if frame_idx > 0:
            trail_pts = np.column_stack((scaled_x[trail_start : frame_idx + 1], scaled_y[trail_start : frame_idx + 1]))
            trail_pts = trail_pts.reshape((-1, 1, 2))

            # Draw trail line
            cv2.polylines(frame_img, [trail_pts], isClosed=False, color=CV2_TRAJECTORY_COLOR, thickness=2)
            # Draw current position dot
            cv2.circle(frame_img, (scaled_x[frame_idx], scaled_y[frame_idx]), radius=4, color=CV2_TRAJECTORY_COLOR, thickness=-1)

        # Check for new crossing
        current_crossing = next((c for c in crossings if c["index"] == frame_idx), None)
        if current_crossing:
            cross_x = int(current_crossing["coordinates"][0] * ratio)
            cross_y = int(current_crossing["coordinates"][1] * ratio)
            text = f"{current_crossing['from']} -> {current_crossing['to']}"
            active_annotations.append({"text": text, "x": cross_x, "y": cross_y, "born": frame_idx})

        active_annotations = [a for a in active_annotations if frame_idx - a["born"] < frames_to_keep_text]

        # Draw Annotations
        for ann in active_annotations:
            # Draw crossing dot
            cv2.circle(frame_img, (ann["x"], ann["y"]), radius=6, color=CV2_CROSSING_COLOR, thickness=-1)

            # Draw text background box
            text_size = cv2.getTextSize(ann["text"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            box_tl = (ann["x"] + 5, ann["y"] - text_size[1] - 10)
            box_br = (ann["x"] + 15 + text_size[0], ann["y"] - 5)
            cv2.rectangle(frame_img, box_tl, box_br, CV2_TEXT_BG_COLOR, -1)
            cv2.rectangle(frame_img, box_tl, box_br, CV2_TEXT_COLOR, 1)

            # Draw text
            cv2.putText(frame_img, ann["text"], (ann["x"] + 10, ann["y"] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CV2_TEXT_COLOR, 1, cv2.LINE_AA)

        video_writer.write(frame_img)

    video_writer.release()


def _matplotlib_animate_maze_crossings(animal: Animal, result: dict, request: AnalysisRequest, polygons: dict) -> None:
    """
    Legacy Matplotlib animation generation. Preserved for reference or fallback.
    """
    console_logger.info(f"Starting animation generation for {animal.id}... This may take a moment.")
    save_folder = request.output_folder
    animal_name = animal.id

    x_pos = result.get("filtered_x").values
    y_pos = result.get("filtered_y").values
    states = result.get("spatial_state_array").values
    fps = request.options.frames_per_second

    if len(x_pos) == 0 or len(states) == 0:
        return

    crossings = []
    previous_state = states[0]
    for i in range(1, len(states)):
        current_state = states[i]
        if current_state != previous_state:
            crossings.append({"index": i, "coordinates": (x_pos[i], y_pos[i]), "from": previous_state, "to": current_state})
            previous_state = current_state

    max_w, max_h = map(int, request.options.max_fig_res)
    img_w, img_h = animal.exp_dimensions()
    ratio = min(max_w / img_w, max_h / img_h) if img_w > 0 and img_h > 0 else 1
    new_size = (int(img_w * ratio / 100), int(img_h * ratio / 100))

    fig, ax = plt.subplots(figsize=new_size, dpi=100)
    ax.set_aspect("equal")
    ax.set_title(f"Body Part Crossing Animation: {animal_name}")

    if animal.image is not None:
        ax.imshow(animal.image)
    else:
        ax.invert_yaxis()

    ax.axis("off")

    for i, (zone_name, poly) in enumerate(polygons.items()):
        px, py = poly.exterior.xy
        color = ANIMATION_POLY_COLORS[i % len(ANIMATION_POLY_COLORS)]
        ax.fill(px, py, alpha=0.3, color=color, label=zone_name)

    (body_part_line,) = ax.plot([], [], "bo-", label="Trajectory", markersize=3)
    (crossing_marker,) = ax.plot([], [], "ro", label="Crossing", markersize=8)

    annotations = [ax.text(0, 0, "", fontsize=8, color="red") for _ in range(2)]
    ax.legend(loc="upper right", fontsize=8)

    def update(frame):
        trail_start = max(0, frame - 15)
        trail_end = frame + 1
        body_part_line.set_data(x_pos[trail_start:trail_end], y_pos[trail_start:trail_end])

        current_crossing = next((c for c in crossings if c["index"] == frame), None)
        if current_crossing:
            crossing_marker.set_data([current_crossing["coordinates"][0]], [current_crossing["coordinates"][1]])
            annotation = ax.annotate(
                f"{current_crossing['from']} --> {current_crossing['to']}",
                xy=current_crossing["coordinates"],
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
            )
            annotations.append(annotation)

            if len(annotations) > 2:
                old_annotation = annotations.pop(0)
                old_annotation.remove()
        else:
            crossing_marker.set_data([], [])

        return [body_part_line, crossing_marker] + annotations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ani = FuncAnimation(fig, update, frames=len(x_pos), interval=1000 / fps, blit=True)
        save_path = os.path.join(save_folder, f"{animal_name} - Validation Video.mp4")

        with tqdm(total=len(x_pos), desc=f"Rendering {animal_name}", unit="frame") as pbar:

            def progress_callback(current_frame, total_frames):
                pbar.update(1)

            try:
                ani.save(save_path, writer="ffmpeg", fps=fps, progress_callback=progress_callback)
            except Exception as e:
                console_logger.error(f"Failed to save animation for {animal_name}. Error: {e}")

    plt.close(fig)
