from __future__ import annotations

import shutil
import tempfile
from behavython.config.defaults import TOTAL_SESSION_STORAGE_QUOTA
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class RuntimeStorageConfig:
    runtime_root: Path
    data_root: Path
    keep_last_sessions: int = TOTAL_SESSION_STORAGE_QUOTA


class RuntimeStorage:
    def __init__(self, config: RuntimeStorageConfig) -> None:
        self.config = config

        self.runtime_root = self.config.runtime_root
        self.data_root = self.config.data_root

        self.logs_root = self.data_root / "logs"
        self.history_root = self.data_root / "history"
        self.cache_root = self.data_root / "cache"

        self._ensure_base_structure()

        self.session_root = self._create_session_root()

        self.session_logs_dir = self.session_root / "logs"
        self.session_dlc_output_dir = self.session_root / "dlc_output"
        self.session_intermediate_dir = self.session_root / "intermediate"
        self.session_debug_dir = self.session_root / "debug"

        self._ensure_session_structure()
        self.prune_old_sessions()

    def _ensure_base_structure(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)

        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.history_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def _create_session_root(self) -> Path:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        unique_suffix = next(tempfile._get_candidate_names())
        session_name = f"session_{timestamp}_{unique_suffix}"

        session_root = self.runtime_root / session_name
        session_root.mkdir(parents=True, exist_ok=False)
        return session_root

    def _ensure_session_structure(self) -> None:
        self.session_logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_dlc_output_dir.mkdir(parents=True, exist_ok=True)
        self.session_intermediate_dir.mkdir(parents=True, exist_ok=True)
        self.session_debug_dir.mkdir(parents=True, exist_ok=True)

    def list_session_dirs(self) -> list[Path]:
        if not self.runtime_root.exists():
            return []

        session_dirs = [
            path
            for path in self.runtime_root.iterdir()
            if path.is_dir() and path.name.startswith("session_")
        ]

        return sorted(session_dirs, key=lambda path: path.stat().st_mtime, reverse=True)

    def prune_old_sessions(self) -> list[Path]:
        keep_last_sessions = max(self.config.keep_last_sessions, 0)
        session_dirs = self.list_session_dirs()

        old_session_dirs = session_dirs[keep_last_sessions:]
        removed_dirs: list[Path] = []

        for session_dir in old_session_dirs:
            if session_dir == self.session_root:
                continue

            shutil.rmtree(session_dir, ignore_errors=True)
            removed_dirs.append(session_dir)

        return removed_dirs