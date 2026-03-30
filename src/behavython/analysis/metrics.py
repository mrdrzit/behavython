import numpy as np
import pandas as pd
from behavython.analysis.primitives import line_trough_triangle_vertex, detect_collision


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
    prev_distances = {}

    for i in runtime:
        A = np.array([nose_x[i], nose_y[i]])
        B = np.array([left_x[i], left_y[i]])
        C = np.array([right_x[i], right_y[i]])

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

            # --- state
            prev_distance = prev_distances.get(idx)
            prev_distances[idx] = distance

            if angle_deg <= 90:
                if prev_distance is not None and distance < prev_distance:
                    state = "approaching"
                else:
                    state = "looking"
            else:
                if prev_distance is not None and distance > prev_distance:
                    state = "retreating"
                else:
                    state = "neutral"

            # --- collision
            collision = detect_collision(
                [Q[0], Q[1]],
                [P[0], P[1]],
                [roi["x"], roi["y"]],
                roi["r"] / 2,
            )

            if collision:
                collision_rows.append(
                    {
                        "collision_flag": 1,
                        "collision_pos": collision,
                        "head_area": head_area,
                        "roi_name": roi["name"],
                        "interaction_state": state,
                        "angle_to_roi": angle_deg,
                        "distance_to_roi": distance,
                        "frame": i,
                    }
                )
            else:
                collision_rows.append(
                    {
                        "collision_flag": 0,
                        "collision_pos": None,
                        "head_area": head_area,
                        "roi_name": None,
                        "interaction_state": "no interaction",
                        "angle_to_roi": angle_deg,
                        "distance_to_roi": distance,
                        "frame": i,
                    }
                )

    collisions_df = pd.DataFrame(collision_rows)

    return {"collisions": collisions_df}
