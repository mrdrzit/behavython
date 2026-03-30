from __future__ import annotations

import numpy as np


def preprocess_animal(animal, request) -> dict:
    """
    Normalize Animal into analysis-ready structure.

    Returns:
        dict with:
            coords, runtime, fps, scaling, roi, analysis_range
    """

    coords = {}

    for bp, values in animal.bodyparts.items():
        coords[bp] = {
            "x": values["x"].to_numpy(),
            "y": values["y"].to_numpy(),
            "likelihood": values["likelihood"].to_numpy(),
        }

    lengths = [len(v["x"]) for v in coords.values()]

    if not lengths:
        raise ValueError("No bodyparts available")

    if len(set(lengths)) != 1:
        raise ValueError("Mismatched frame lengths across bodyparts")

    n_frames = lengths[0]
    fps = request.options.frames_per_second
    max_time = request.options.task_duration
    threshold = request.options.threshold
    total_frames = int(max_time * fps)
    trim_seconds = request.options.trim_amount
    trim_frames = int(trim_seconds * fps)
    crop_video = request.options.crop_video

    if crop_video:
        start = trim_frames
        end = min(trim_frames + total_frames, n_frames)
    else:
        start = 0
        end = min(total_frames, n_frames)

    if start >= end:
        raise ValueError("Invalid runtime range (trim too large)")

    runtime = np.arange(start, end)
    arena_w = request.options.arena_width
    arena_h = request.options.arena_height
    video_h, video_w = animal.exp_dimensions()
    scale_x = arena_w / video_w if arena_w and video_w else 1.0
    scale_y = arena_h / video_h if arena_h and video_h else 1.0

    scaling = {
        "x": scale_x,
        "y": scale_y,
    }

    roi_data = []

    if hasattr(animal, "rois") and animal.rois:
        for roi in animal.rois:
            if not roi.get("x"):
                continue

            roi_data.append(
                {
                    "x": roi["x"],
                    "y": roi["y"],
                    "r": (roi["width"] + roi["height"]) / 2,
                    "name": roi.get("file", "roi"),
                }
            )

    analysis_range = (start, end)

    return {
        "coords": coords,  # canonical bodyparts
        "runtime": runtime,  # np.array of frame indices
        "fps": fps,
        "n_frames": n_frames,
        "scaling": scaling,
        "threshold": threshold,
        "roi": roi_data,
        "analysis_range": analysis_range,
    }
