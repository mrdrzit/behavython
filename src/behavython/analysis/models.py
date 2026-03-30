from __future__ import annotations

from dataclasses import dataclass

@dataclass(slots=True)
class analysis_options:
    arena_width: int
    arena_height: int
    frames_per_second: int
    experiment_type: str
    max_fig_res: list[str]
    algo_type: str
    threshold: float
    task_duration: int
    trim_amount: int
    crop_video: bool
    plot_options: str


@dataclass(slots=True)
class analysis_request:
    input_files: list[str]
    output_folder: str
    options: analysis_options
