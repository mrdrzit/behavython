from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class VideoInputSource:
    folder_path: str = ""
    txt_path: str = ""


@dataclass(slots=True)
class AnalysisInputSource:
    selected_files: list[str]


@dataclass(slots=True)
class OutputFolderSource:
    selected_folder: str = ""


@dataclass(slots=True)
class ResolvedVideoInput:
    paths: list[str] = field(default_factory=list)
    source_kind: str | None = None
    skipped_entries: list[str] = field(default_factory=list)
    duplicate_entries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_paths(self) -> bool:
        return bool(self.paths)


@dataclass(slots=True)
class ResolvedAnalysisInput:
    paths: list[str] = field(default_factory=list)
    skipped_entries: list[str] = field(default_factory=list)
    duplicate_entries: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_paths(self) -> bool:
        return bool(self.paths)


@dataclass(slots=True)
class ResolvedOutputFolder:
    path: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def has_path(self) -> bool:
        return bool(self.path)