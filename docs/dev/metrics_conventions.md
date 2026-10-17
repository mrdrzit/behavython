# Metrics Conventions

How behavython turns tracking data into metrics today. Behavython exports per-animal metrics; statistical testing happens downstream, outside this package.

**Core rule:** a change to any threshold, window, or definition below changes results users may already have published. Such changes need an explicit note in the release notes or README, and the old behavior should stay reproducible when practical.

## 1. Pipelines
| Experiment types | Entry point | Primary bodypart |
| :--- | :--- | :--- |
| `social_recognition`, `social_discrimination`, `object_discrimination` | `workflow.analyze_animal` | `center` (movement), `nose` + ears (interaction, heatmap) |
| `open_field`, `elevated_plus_maze` | `workflow.analyze_maze_animal` | `center` (via `get_primary_tracking_data`) |

Input positions are DeepLabCut `*filtered.csv` files (`ANALYSIS_REQUIRED_SUFFIXES` in `core/defaults.py`). Bodypart names are normalized through `BODYPART_MAPPING` to `CANONICAL_BODYPARTS`.

## 2. Analysis Window
- `crop_video=True`: frames `[trim * fps, trim * fps + duration * fps)`.
- `crop_video=False`: frames `[0, duration * fps)`.
- If the video is shorter, the window is clamped and a warning is appended to `animal.logs`.
- Every metric for an animal uses the same window.
- This logic is duplicated in `metrics.preprocess_animal` and `workflow.analyze_maze_animal`. Change both together.

## 3. Pixel-to-cm Scaling
- Interaction experiments: `arena_width / video_width` and `arena_height / video_height` (the whole frame is treated as the arena).
- Open field: arena size divided by the pixel distance between arena corners.
- Elevated plus maze: arena size divided by the pixel distance between arm extremities (`maze_points[11]`-`[4]` closed, `[2]`-`[7]` open).
- Falls back to `1.0` (pixels) when sizes are missing.

## 4. Movement (`compute_movement_metrics`)
- Per-frame displacement of `center` in cm.
- Displacement below the noise threshold is set to 0. Threshold depends on the animal: mouse `0.0267`, rat `0.0667` cm/frame, from `MOVEMENT_THRESHOLD_CM_BY_ANIMAL` in `core/defaults.py` (used by both GUI and CLI).
- Displacement above 55 cm/frame is set to 0 (tracking jump).
- `time_moving_s` = frames with displacement > 0 divided by fps; `time_resting_s` is the complement.
- `mean_velocity_cm_s` = mean of per-frame displacement / dt.
- `filters.apply_rolling_window_filter` (jump filter + cubic spline) exists but is disabled in the maze workflow.
- Likelihood values are loaded but not used by the metrics; filtering is expected to come from DLC.

## 5. ROI Interaction (`compute_roi_interaction`)
- Thresholds are normalized by median body length: nose-tail, else 2 x nose-center, else 4 x nose-to-ear-midpoint, else 100 px.
- Gaze cone uses hysteresis: enter below `ANGLE_CONE_ENTER` (45 deg), exit above `ANGLE_CONE_EXIT` (60 deg).
- Speed and distance deltas use a `ROI_DELTA_LOOKBACK_FRAMES` (3) buffer.
- States: `approaching`, `retreating`, `looking`, `neutral`, `collision`. Locomotion gate `LOCOMOTION_THRESHOLD_BL` (0.02 BL/frame); approach gate `APPROACH_THRESHOLD_BL` (0.1 BL/frame).
- Collision = gaze line intersects a circle at the ROI center with radius `r / 2`, where `r = (width + height) / 2`. Collision overrides the other states.
- `exploration_time_s` = collision frames / fps. Left/right split matches ROI names containing `roiL` / `roiR`.

## 6. Social Bouts (`compute_social_behavior_metrics`)
- Events are contiguous blocks of at least `MIN_EVENT_FRAMES` (3) frames, computed per ROI.
- Each approach is linked to the next collision, then retreat, within `LINK_WINDOW_S` (1.0 s, converted to frames with the video fps). Outcome: `collision`, `abortive_retreat`, or `approach_only`.
- `MIN_EVENT_FRAMES` and `ROI_DELTA_LOOKBACK_FRAMES` are still in frames.
- Inter-bout intervals only count positive gaps.

## 7. Zones and Crossings (maze experiments)
- Each frame gets the first polygon from `geometry.py` that contains the point; `none` if outside, `missing` if NaN.
- `time_in_zones_s` = frames per zone / fps.
- Open field is a 3 x 3 grid; `zone_r1_c1` is the center, all other `zone_*` cells are border.
- Crossing keys per experiment type are fixed in `compute_crossings`; renaming them changes export columns.

## 8. Naming and Export
- Result dict keys: snake_case with a unit suffix (`_s`, `_cm`, `_cm_s`); per-frame arrays end in `_array`.
- `export_summary_metrics` writes `analysis_summary.csv` / `.xlsx`, metrics as rows and animals as columns, with human labels in the form `name (unit)`.
- `export_results_to_parquet` writes `analysis_summary.parquet`, `analysis_timeseries.parquet`, and `analysis_collisions.parquet`.
- A new metric must be added to the result dict, both exporters, and the README outputs section.
