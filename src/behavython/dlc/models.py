from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DLCVideoAnalysisRequest:
    config_path: str
    video_paths: list[str]
    create_plots: bool = True


@dataclass(slots=True)
class DLCSkeletonExtractionRequest:
    config_path: str
    video_paths: list[str]


@dataclass(slots=True)
class DLCFrameExtractionRequest:
    video_paths: list[str]
    override_frame_number: int | None = None


@dataclass(slots=True)
class DLCClearUnusedFilesRequest:
    folder_path: str
    task_type: str
