import numpy as np
import pandas as pd
from shapely.geometry import Point
from scipy import stats
from collections import deque
from behavython.pipeline.geometry import line_trough_triangle_vertex, detect_collision, create_frequency_grid
from behavython.core.defaults import ANGLE_CONE_ENTER, ANGLE_CONE_EXIT, ROI_DELTA_LOOKBACK_FRAMES, APPROACH_THRESHOLD_BL, LOCOMOTION_THRESHOLD_BL
from behavython.core.exceptions import AnalysisError


def compute_roi_interaction(data) -> dict:
    coords = data["coords"]
    runtime = data["runtime"]
    roi_list = data["roi"]

    nose_x = coords["nose"]["x"]
    nose_y = coords["nose"]["y"]
    left_x = coords["left_ear"]["x"]
    left_y = coords["left_ear"]["y"]
    right_x = coords["right_ear"]["x"]
    right_y = coords["right_ear"]["y"]

    collision_rows = []
    distance_windows = {}
    in_cone = {}
    center_window = deque(maxlen=ROI_DELTA_LOOKBACK_FRAMES)

    # --- calculate body length for normalized thresholds ---
    # 1. Try Full Body (Nose to Tail)
    if "tail" in coords.keys():
        dist = np.sqrt((nose_x[runtime] - coords["tail"]["x"][runtime]) ** 2 + (nose_y[runtime] - coords["tail"]["y"][runtime]) ** 2)
        median_body_length = np.nanmedian(dist)
    # 2. Try Half Body (Nose to Center)
    elif "center" in coords.keys():
        dist = np.sqrt((nose_x[runtime] - coords["center"]["x"][runtime]) ** 2 + (nose_y[runtime] - coords["center"]["y"][runtime]) ** 2)
        median_body_length = np.nanmedian(dist) * 2.0
    # 3. Guaranteed Fallback: Head Length (Nose to Ears)
    else:
        ear_mid_x = (left_x[runtime] + right_x[runtime]) / 2.0
        ear_mid_y = (left_y[runtime] + right_y[runtime]) / 2.0
        head_dist = np.sqrt((nose_x[runtime] - ear_mid_x) ** 2 + (nose_y[runtime] - ear_mid_y) ** 2)
        median_body_length = np.nanmedian(head_dist) * 4.0

    # Ultimate safety fallback in case all coordinates were NaNs
    if np.isnan(median_body_length) or median_body_length == 0:
        median_body_length = 100.0

    for i in runtime:
        A = np.array([nose_x[i], nose_y[i]])
        B = np.array([left_x[i], left_y[i]])
        C = np.array([right_x[i], right_y[i]])

        # --- center locomotion speed ---
        if "center" in coords.keys():
            c_pt = np.array([coords["center"]["x"][i], coords["center"]["y"][i]])
        else:
            c_pt = A  # fallback to nose

        center_window.append(c_pt)
        if len(center_window) >= 2:
            oldest_c = center_window[0]
            center_displacement = np.linalg.norm(c_pt - oldest_c)
            center_speed_bl = (center_displacement / (len(center_window) - 1)) / median_body_length
        else:
            center_speed_bl = 0.0

        # --- head triangle area (Heron)
        s1 = np.linalg.norm(B - A)
        s2 = np.linalg.norm(C - B)
        s3 = np.linalg.norm(A - C)
        s = (s1 + s2 + s3) / 2
        head_area = np.sqrt(max(s * (s - s1) * (s - s2) * (s - s3), 0))

        # --- gaze line
        P, Q = line_trough_triangle_vertex(A, B, C)

        for idx, roi in enumerate(roi_list):
            T = np.array([roi["x"], roi["y"]])

            # vectors
            v_gaze = np.array(Q) - np.array(P)
            v_target = T - A

            # normalize safely
            norm_gaze = np.linalg.norm(v_gaze)
            norm_target = np.linalg.norm(v_target)

            if norm_gaze == 0 or norm_target == 0:
                angle_deg = 180.0
            else:
                v_gaze_n = v_gaze / norm_gaze
                v_target_n = v_target / norm_target

                cos_theta = np.clip(np.dot(v_gaze_n, v_target_n), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(cos_theta))

            distance = norm_target

            # --- calculate delta distance & state ---
            distance_window = distance_windows.setdefault(idx, deque(maxlen=ROI_DELTA_LOOKBACK_FRAMES))
            distance_window.append(distance)

            if len(distance_window) >= 2:
                oldest_distance = distance_window[0]
                frame_span = len(distance_window) - 1
                # Keep delta in a per-frame scale while using a short temporal buffer.
                delta_distance = (distance - oldest_distance) / frame_span
            else:
                delta_distance = 0.0

            # --- per-ROI cone hysteresis ---
            currently_in_cone = in_cone.get(idx, False)
            if currently_in_cone:
                # Once in cone, only leave after crossing EXIT threshold
                if angle_deg > ANGLE_CONE_EXIT:
                    currently_in_cone = False
            else:
                # Once out of cone, only enter after crossing ENTER threshold
                if angle_deg < ANGLE_CONE_ENTER:
                    currently_in_cone = True
            in_cone[idx] = currently_in_cone

            # --- classify directional state using hysteresis cone gate ---
            normalized_delta = delta_distance / median_body_length

            if center_speed_bl >= LOCOMOTION_THRESHOLD_BL:
                if currently_in_cone:
                    state = "approaching" if normalized_delta < APPROACH_THRESHOLD_BL else "looking"
                else:
                    state = "retreating" if normalized_delta > APPROACH_THRESHOLD_BL else "neutral"
            else:
                state = "looking" if currently_in_cone else "neutral"

            # --- collision ---
            collision = detect_collision(
                [Q[0], Q[1]],
                [P[0], P[1]],
                [roi["x"], roi["y"]],
                roi["r"] / 2,
            )

            # Collision is mutually exclusive with directional interaction states.
            # If a collision is present, force state to "collision" so downstream
            # counts do not classify the same frame as approaching/retreating/looking.
            final_state = "collision" if collision else state

            collision_rows.append(
                {
                    "frame": i,
                    "roi_name": roi["name"],
                    "interaction_state": final_state,
                    "collision_flag": 1 if collision else 0,
                    "collision_pos": collision if collision else None,
                    "head_area": head_area,
                    "angle_to_roi": angle_deg,
                    "distance_to_roi": distance,
                    "delta distance": delta_distance,  # Required by your extract_approach.py
                    "in_cone": currently_in_cone,
                }
            )

    collisions_df = pd.DataFrame(collision_rows)

    return {"collisions": collisions_df}


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
        raise AnalysisError("No bodyparts available")

    if len(set(lengths)) != 1:
        raise AnalysisError("Mismatched frame lengths across bodyparts")

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
        end = trim_frames + total_frames

        if start >= n_frames:
            raise AnalysisError(f"Trim amount ({trim_seconds}s) exceeds the entire video length ({n_frames / fps:.1f}s).")

        if end > n_frames:
            animal.logs.append(
                {
                    "level": "warning",
                    "message": f"Requested {max_time}s after {trim_seconds}s trim, but video is only {n_frames / fps:.1f}s. Analyzing remaining {(n_frames - start) / fps:.1f}s.",
                    "context": "analysis_range",
                }
            )
            end = n_frames
    else:
        start = 0
        if total_frames > n_frames:
            animal.logs.append(
                {
                    "level": "warning",
                    "message": f"Requested {max_time}s analysis, but video is only {n_frames / fps:.1f}s. Analyzing entire video.",
                    "context": "analysis_range",
                }
            )
        end = min(total_frames, n_frames)

    if start >= end:
        raise AnalysisError("Invalid runtime range (start >= end).")

    runtime = np.arange(start, end)
    arena_w = request.options.arena_width
    arena_h = request.options.arena_height
    video_w, video_h = animal.exp_dimensions()
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
                    "name": roi.get("name", "roi"),
                    "x": roi["x"],
                    "y": roi["y"],
                    "r": (roi["width"] + roi["height"]) / 2,
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


def compute_movement_metrics(
    centro_x: np.ndarray, centro_y: np.ndarray, fps: float, scale_x: float, scale_y: float, threshold: float = 0.0667
) -> dict:
    """
    Calculates displacement, velocity, acceleration, and temporal states.
    Expects arrays that have already been sliced to the analysis range and filtered.
    """
    # Apply scaling to the already filtered/sliced coordinates
    scaled_x = centro_x * scale_x
    scaled_y = centro_y * scale_y

    # Calculate step differences
    d_x = np.append(0, np.diff(scaled_x))
    d_y = np.append(0, np.diff(scaled_y))

    # Raw displacement
    displacement = np.sqrt(np.square(d_x) + np.square(d_y))

    # Apply thresholds (noise filtering and anomaly removal)
    displacement[displacement < threshold] = 0
    displacement[displacement > 55] = 0  # Max physical displacement boundary

    # Total distance calculation
    accumulate_distance = np.cumsum(displacement)
    total_distance = np.max(accumulate_distance) if len(accumulate_distance) > 0 else 0.0

    # Time vector for derivatives
    time_vector = np.linspace(0, len(scaled_x) / fps, len(scaled_x))
    dt = np.append(0, np.diff(time_vector))

    # Safely calculate velocity and acceleration, ignoring division by zero during resting states
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = np.divide(displacement, dt)
        velocity[~np.isfinite(velocity)] = 0  # Clean up NaNs or Infs
        mean_velocity = np.nanmean(velocity) if len(velocity) > 0 else 0.0

        dv = np.append(0, np.diff(velocity))
        acceleration = np.divide(dv, dt)
        acceleration[~np.isfinite(acceleration)] = 0

    # Temporal states
    time_moving = np.sum(displacement > 0) * (1 / fps)
    time_resting = np.sum(displacement == 0) * (1 / fps)
    movements_count = np.sum(displacement > 0)

    return {
        "total_distance_cm": float(total_distance),
        "mean_velocity_cm_s": float(mean_velocity),
        "time_moving_s": float(time_moving),
        "time_resting_s": float(time_resting),
        "movements_count": int(movements_count),
        "velocity_array": velocity,
        "acceleration_array": acceleration,
        "displacement_array": displacement,
        "accumulate_distance_array": accumulate_distance,
    }


def compute_spatial_metrics(data: dict, movement_metrics: dict) -> dict:
    """
    Calculates all spatial metrics including frequency grids (heatmaps),
    spatial density (KDE), and path smoothing for downstream visualization.
    """
    coords = data["coords"]
    start, end = data["analysis_range"]
    analysis_range = data["analysis_range"]
    bin_size = 10

    nose_bp = coords["nose"] if "nose" in coords else None

    if nose_bp is None:
        raise AnalysisError("Missing 'nose' bodypart required for spatial metrics.")

    position_grid = create_frequency_grid(x_values=nose_bp["x"], y_values=nose_bp["y"], bin_size=bin_size, analysis_range=analysis_range)

    # Base coordinates for KDE/Smoothing
    x_axe = nose_bp["x"][start:end]
    y_axe = nose_bp["y"][start:end]
    kde_space_coordinates = np.vstack([x_axe, y_axe])

    # KDE Density Map
    try:
        kde_instance = stats.gaussian_kde(kde_space_coordinates)
        point_density_function = kde_instance.evaluate(kde_space_coordinates)

        min_pdf = np.min(point_density_function)
        max_pdf = np.max(point_density_function)

        if max_pdf > min_pdf:
            color_limits = (point_density_function - min_pdf) / (max_pdf - min_pdf)
        else:
            color_limits = np.zeros_like(point_density_function)

    except np.linalg.LinAlgError:
        # Fallback for singular matrix errors
        point_density_function = np.zeros_like(x_axe)
        color_limits = np.zeros_like(x_axe)

    # Path Smoothing
    movement_points = np.column_stack([x_axe, y_axe]).reshape(-1, 1, 2)
    movement_segments = np.concatenate([movement_points[:-1], movement_points[1:]], axis=1)
    filter_size = 4
    moving_average_filter = np.ones((filter_size,)) / filter_size
    smooth_segs = np.apply_along_axis(lambda m: np.convolve(m, moving_average_filter, mode="same"), axis=0, arr=movement_segments)

    return {
        "position_grid": position_grid,
        "kde_density_array": point_density_function,
        "color_limits_array": color_limits,
        "smooth_segments_array": smooth_segs,
        "raw_x_pos_array": x_axe,
        "raw_y_pos_array": y_axe,
    }


def compute_exploration_metrics(interaction_data: dict, fps: float) -> dict:
    """
    Calculates total and ROI-specific exploration times based on collision data.
    """
    collisions_df = interaction_data.get("collisions")

    # Handle cases where there are no ROIs or no collision data
    if collisions_df is None or collisions_df.empty:
        return {
            "exploration_time_s": 0.0,
            "exploration_time_right_s": 0.0,
            "exploration_time_left_s": 0.0,
        }

    # Calculate total exploration time
    exploration_mask = collisions_df["collision_flag"] > 0
    exploration_mask = exploration_mask.astype(int)
    exploration_time = np.sum(exploration_mask) * (1 / fps)

    # Calculate exploration time for specific ROIs
    # Note: str.contains is used here to match the original logic.
    roi_names = collisions_df["roi_name"].fillna("")

    # We combine the exploration mask with the name filter to ensure we only count
    # frames where a collision ACTUALLY happened with that specific ROI.
    count_right = (exploration_mask & roi_names.str.contains("roiR")).sum()
    count_left = (exploration_mask & roi_names.str.contains("roiL")).sum()

    exploration_time_right = float(count_right * (1 / fps))
    exploration_time_left = float(count_left * (1 / fps))

    return {
        "exploration_time_s": exploration_time,
        "exploration_time_right_s": exploration_time_right,
        "exploration_time_left_s": exploration_time_left,
    }


def extract_collision_coordinates(interaction_data: dict) -> dict:
    collisions_df = interaction_data["collisions"]

    if collisions_df.empty:
        return {
            "x_collision_data": np.array([]),
            "y_collision_data": np.array([]),
        }

    valid = collisions_df["collision_flag"] > 0
    positions = collisions_df.loc[valid, "collision_pos"].dropna()

    if positions.empty:
        return {
            "x_collision_data": np.array([]),
            "y_collision_data": np.array([]),
        }

    coords = np.vstack(positions.to_numpy())

    return {
        "x_collision_data": coords[:, 0],
        "y_collision_data": coords[:, 1],
    }


def compute_zone_metrics(filtered_x: np.ndarray, filtered_y: np.ndarray, geometry_dict: dict, fps: float) -> dict:
    """
    Evaluates coordinates against maze geometries to tag spatial states (zones)
    and calculates the total time spent in each zone.

    This fully replaces the legacy `analyze_entries` and `analyze_time_in_each_sector`.
    """
    states = []

    # Tag each frame with a zone
    for x, y in zip(filtered_x, filtered_y):
        if pd.isna(x) or pd.isna(y):
            states.append("missing")
            continue

        point = Point(x, y)
        current_zone = "none"  # Default fallback matching the legacy outlier/outside tag

        # We iterate through the flat geometry dictionary provided by geometry.py
        for zone_name, polygon in geometry_dict.items():
            if polygon.contains(point):
                current_zone = zone_name
                break

        states.append(current_zone)

    states_series = pd.Series(states, name="spatial_state")

    # Calculate time in zones using highly optimized Pandas value_counts
    time_per_frame = 1.0 / fps
    frame_counts = states_series.value_counts()

    time_in_zones = {}
    for zone, count in frame_counts.items():
        time_in_zones[zone] = float(count * time_per_frame)

    return {"spatial_state_array": states_series, "time_in_zones_s": time_in_zones}


def compute_crossings(state_array: pd.Series, experiment_type: str) -> dict:
    """
    Calculates zone transitions and maps them to the legacy dictionary keys.
    Replaces the frame-by-frame loop with vectorized array shifts.
    """
    previous_states = state_array.shift(1)

    # Keep only rows where the state actually changed
    transitions = pd.DataFrame({"from_zone": previous_states, "to_zone": state_array}).dropna()
    transitions = transitions[transitions["from_zone"] != transitions["to_zone"]]

    counts = {}

    if experiment_type == "plus_maze":
        counts = {
            "to_center_from_top_open": 0,
            "to_center_from_right_closed": 0,
            "to_center_from_bottom_open": 0,
            "to_center_from_left_closed": 0,
            "to_top_open_from_center": 0,
            "to_right_closed_from_center": 0,
            "to_bottom_open_from_center": 0,
            "to_left_closed_from_center": 0,
            "crossings_from_outside_to_maze": 0,
            "crossings_from_maze_to_outside": 0,
            "crossing_from_none": 0,
            "unrecognized_crossings": 0,
        }

        for _, row in transitions.iterrows():
            f_z, t_z = row["from_zone"], row["to_zone"]

            if f_z == "none" or f_z == "missing":
                if t_z in ["none", "missing"]:
                    continue
                counts["crossing_from_none"] += 1
                counts["crossings_from_outside_to_maze"] += 1
            elif f_z == "top_open" and t_z == "center":
                counts["to_center_from_top_open"] += 1
            elif f_z == "center" and t_z == "top_open":
                counts["to_top_open_from_center"] += 1
            elif f_z == "right_closed" and t_z == "center":
                counts["to_center_from_right_closed"] += 1
            elif f_z == "center" and t_z == "right_closed":
                counts["to_right_closed_from_center"] += 1
            elif f_z == "bottom_open" and t_z == "center":
                counts["to_center_from_bottom_open"] += 1
            elif f_z == "center" and t_z == "bottom_open":
                counts["to_bottom_open_from_center"] += 1
            elif f_z == "left_closed" and t_z == "center":
                counts["to_center_from_left_closed"] += 1
            elif f_z == "center" and t_z == "left_closed":
                counts["to_left_closed_from_center"] += 1
            elif t_z == "none" or t_z == "missing":
                counts["crossings_from_maze_to_outside"] += 1
            else:
                counts["unrecognized_crossings"] += 1

    elif experiment_type == "open_field":
        counts = {"to_center": 0, "to_border": 0, "from_or_to_outside": 0, "no_crossing": 0, "crossing_from_none": 0}

        # Helper to categorize the 3x3 grid into 'center' and 'border'
        def get_of_category(z):
            if z in ["none", "missing"]:
                return "none"
            if z == "zone_r1_c1":
                return "center"  # Exact center of the 3x3 grid
            if z.startswith("zone_"):
                return "border"
            return "unknown"

        for _, row in transitions.iterrows():
            f_cat = get_of_category(row["from_zone"])
            t_cat = get_of_category(row["to_zone"])

            if f_cat == "none" and t_cat == "none":
                continue
            elif f_cat == "none":
                counts["crossing_from_none"] += 1
                counts["from_or_to_outside"] += 1
            elif t_cat == "none":
                counts["from_or_to_outside"] += 1
            elif f_cat == "center" and t_cat == "border":
                counts["to_border"] += 1
            elif f_cat == "border" and t_cat == "center":
                counts["to_center"] += 1
            elif f_cat == t_cat:
                counts["no_crossing"] += 1
            else:
                counts["from_or_to_outside"] += 1

    return counts


def compute_social_behavior_metrics(collisions_df: pd.DataFrame, fps: float, min_frames: int = 3, link_window: int = 30) -> dict:
    """
    Extracts bout counts, mean inter-bout intervals, and outcome proportions
    (investigation, approach, abortive) based on the social behavior dissertation logic.
    """
    if collisions_df.empty:
        return {
            "total_bouts": 0,
            "mean_inter_bout_interval_s": 0.0,
            "investigation_proportion": 0.0,
            "approach_proportion": 0.0,
            "abortive_retreat_proportion": 0.0,
        }

    sequences = []

    # 1. Group by ROI to prevent .shift() from mixing interleaved ROI rows
    for roi_name, roi_df in collisions_df.groupby("roi_name"):
        # Skip the 'None' fallback if the animal is interacting with nothing
        if pd.isna(roi_name):
            continue

        # Ensure correct temporal ordering for this specific ROI
        roi_df = roi_df.sort_values("frame").reset_index(drop=True)

        # Helper to extract valid contiguous blocks
        def get_blocks(condition):
            blocks = condition.ne(condition.shift()).cumsum()
            valid_blocks = []
            for _, block in roi_df[condition].groupby(blocks):
                if len(block) >= min_frames:
                    valid_blocks.append({"start_frame": block["frame"].iloc[0], "end_frame": block["frame"].iloc[-1]})
            return valid_blocks

        # 2. Extract Events
        approaches = get_blocks(roi_df["interaction_state"] == "approaching")
        collisions = get_blocks(roi_df["collision_flag"] == 1)
        retreats = get_blocks(roi_df["interaction_state"] == "retreating")

        used_collisions = set()
        used_retreats = set()

        # 3. Build Sequences
        for a in approaches:
            a_start, a_end = a["start_frame"], a["end_frame"]

            # ---- Find Collision ----
            linked_col = next(
                (
                    c
                    for i, c in enumerate(collisions)
                    if i not in used_collisions and c["start_frame"] > a_start and c["start_frame"] <= a_end + link_window
                ),
                None,
            )
            if linked_col:
                used_collisions.add(collisions.index(linked_col))

            ref_end = linked_col["end_frame"] if linked_col else a_end

            # ---- Find Retreat ----
            linked_ret = next(
                (
                    r
                    for i, r in enumerate(retreats)
                    if i not in used_retreats and r["start_frame"] > ref_end and r["start_frame"] <= ref_end + link_window
                ),
                None,
            )
            if linked_ret:
                used_retreats.add(retreats.index(linked_ret))

            # ---- Determine Outcome ----
            if linked_col:
                outcome = "collision"
            elif linked_ret:
                outcome = "abortive_retreat"
            else:
                outcome = "approach_only"

            seq_end = linked_ret["end_frame"] if linked_ret else (linked_col["end_frame"] if linked_col else a_end)

            sequences.append({"start_frame": a_start, "end_frame": seq_end, "outcome": outcome})

    # 4. Global Aggregation (Combine sequences from all ROIs)
    total_bouts = len(sequences)
    if total_bouts == 0:
        return {
            "total_bouts": 0,
            "mean_inter_bout_interval_s": 0.0,
            "investigation_proportion": 0.0,
            "approach_proportion": 0.0,
            "abortive_retreat_proportion": 0.0,
            "collision_bouts": 0,
            "approach_only_bouts": 0,
            "abortive_retreat_bouts": 0,
            "mean_interval_collision_s": 0.0,
            "mean_interval_approach_only_s": 0.0,
            "mean_interval_abortive_retreat_s": 0.0,
        }

    outcomes = [s["outcome"] for s in sequences]
    n_col = outcomes.count("collision")
    n_app = outcomes.count("approach_only")
    n_abt = outcomes.count("abortive_retreat")

    def get_mean_interval(seq_list):
        """Helper to calculate the mean gap in seconds for a given list of sequences."""
        intervals_s = []
        for i in range(len(seq_list) - 1):
            gap_frames = seq_list[i + 1]["start_frame"] - seq_list[i]["end_frame"]
            if gap_frames > 0:
                intervals_s.append(gap_frames / fps)
        return float(np.mean(intervals_s)) if intervals_s else 0.0

    # Overall interval
    overall_mean_interval = get_mean_interval(sequences)

    # State-specific intervals
    col_seqs = [s for s in sequences if s["outcome"] == "collision"]
    app_seqs = [s for s in sequences if s["outcome"] == "approach_only"]
    abt_seqs = [s for s in sequences if s["outcome"] == "abortive_retreat"]

    return {
        # Overall
        "total_bouts": total_bouts,
        "mean_inter_bout_interval_s": overall_mean_interval,
        # Proportions
        "investigation_proportion": float(n_col / total_bouts),
        "approach_proportion": float(n_app / total_bouts),
        "abortive_retreat_proportion": float(n_abt / total_bouts),
        # Specific Counts
        "collision_bouts": n_col,
        "approach_only_bouts": n_app,
        "abortive_retreat_bouts": n_abt,
        # Specific Intervals
        "mean_interval_collision_s": get_mean_interval(col_seqs),
        "mean_interval_approach_only_s": get_mean_interval(app_seqs),
        "mean_interval_abortive_retreat_s": get_mean_interval(abt_seqs),
    }
