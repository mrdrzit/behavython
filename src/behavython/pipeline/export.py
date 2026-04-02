import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Callable


def export_results_to_parquet(results: list[dict], output_folder: str) -> None:
    """
    Separates flat summary metrics and time-series arrays into distinct DataFrames
    and exports them as Parquet files.
    """
    summary_rows = []
    timeseries_frames = []

    for res in results:
        animal_id = res["animal_id"]

        # 1. Extract Summary Metrics (Scalars)
        # These are used for statistical aggregation and Excel reports
        summary_rows.append(
            {
                "animal_id": animal_id,
                "experiment_type": res.get("experiment_type"),
                "total_distance_cm": res.get("total_distance_cm"),
                "mean_velocity_cm_s": res.get("mean_velocity_cm_s"),
                "time_moving_s": res.get("time_moving_s"),
                "time_resting_s": res.get("time_resting_s"),
                "exploration_time_s": res.get("exploration_time_s"),
                "exploration_time_right_s": res.get("exploration_time_right_s"),
                "exploration_time_left_s": res.get("exploration_time_left_s"),
            }
        )

        # 2. Extract Time-Series Data (Arrays)
        # These are used for downstream plotting and visualization
        if "velocity_array" in res:
            ts_df = pd.DataFrame(
                {
                    "animal_id": animal_id,
                    "frame": range(len(res["velocity_array"])),
                    "raw_x": res.get("raw_x_pos_array", []),
                    "raw_y": res.get("raw_y_pos_array", []),
                    "velocity": res.get("velocity_array", []),
                    "acceleration": res.get("acceleration_array", []),
                    "displacement": res.get("displacement_array", []),
                }
            )
            timeseries_frames.append(ts_df)

    # Convert to DataFrames
    summary_df = pd.DataFrame(summary_rows)

    # Write Summary Parquet
    summary_path = os.path.join(output_folder, "analysis_summary.parquet")
    summary_table = pa.Table.from_pandas(summary_df)
    pq.write_table(summary_table, summary_path)

    # Write Time-Series Parquet
    if timeseries_frames:
        full_ts_df = pd.concat(timeseries_frames, ignore_index=True)
        ts_path = os.path.join(output_folder, "analysis_timeseries.parquet")
        ts_table = pa.Table.from_pandas(full_ts_df)
        pq.write_table(ts_table, ts_path)


def export_summary_metrics(results: list[dict], output_folder: str, log: Callable | None = None) -> None:
    """
    Filters metrics based on experiment type, transposes them (metrics as rows,
    animals as columns), and exports to CSV and Excel.
    """
    animal_frames = []

    for res in results:
        exp_type = res.get("experiment_type")
        animal_id = res.get("animal_id")

        if exp_type == "njr":
            metrics = {
                "exploration_time (s)": res.get("exploration_time_s", 0),
                "exploration_time_right (s)": res.get("exploration_time_right_s", 0),
                "exploration_time_left (s)": res.get("exploration_time_left_s", 0),
                "time moving (s)": res.get("time_moving_s", 0),
                "time resting (s)": res.get("time_resting_s", 0),
                "total distance (cm)": res.get("total_distance_cm", 0),
                "mean velocity (cm/s)": res.get("mean_velocity_cm_s", 0),
            }
        elif exp_type == "social_recognition":
            metrics = {
                "exploration_time (s)": res.get("exploration_time_s", 0),
                "time moving (s)": res.get("time_moving_s", 0),
                "time resting (s)": res.get("time_resting_s", 0),
                "total distance (cm)": res.get("total_distance_cm", 0),
                "mean velocity (cm/s)": res.get("mean_velocity_cm_s", 0),
            }
        else:
            # Safe fallback for any unspecified experiment types
            metrics = {
                "time moving (s)": res.get("time_moving_s", 0),
                "time resting (s)": res.get("time_resting_s", 0),
                "total distance (cm)": res.get("total_distance_cm", 0),
                "mean velocity (cm/s)": res.get("mean_velocity_cm_s", 0),
            }

        # 2. Create the DataFrame and Transpose it (.T)
        # This makes the index the metric names, and the column name the animal_id
        df = pd.DataFrame(metrics, index=[animal_id]).T
        animal_frames.append(df)

    if not animal_frames:
        return

    # 3. Join all individual animal DataFrames side-by-side
    # pd.concat with axis=1 aligns the rows (metrics) automatically
    final_df = pd.concat(animal_frames, axis=1)

    # Name the index column for cleaner output
    final_df.index.name = "Metric"

    # 4. Export to Disk
    csv_path = os.path.join(output_folder, "analysis_summary.csv")
    xlsx_path = os.path.join(output_folder, "analysis_summary.xlsx")

    final_df.to_csv(csv_path)

    try:
        final_df.to_excel(xlsx_path)
    except ModuleNotFoundError:
        if log:
            log.emit("warning", "Excel export failed: 'openpyxl' not found")
