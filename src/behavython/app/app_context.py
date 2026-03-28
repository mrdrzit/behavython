from __future__ import annotations

from dataclasses import dataclass, field
from PySide6.QtCore import QThreadPool
from behavython.config.paths import DATA_ROOT, RUNTIME_ROOT
from behavython.services.app_logging import AppLoggingService
from behavython.services.runtime_storage import RuntimeStorage, RuntimeStorageConfig


@dataclass
class AppContext:
    threadpool: QThreadPool = field(default_factory=QThreadPool)
    debug_mode: bool = False
    runtime_storage: RuntimeStorage = field(init=False)
    app_logging: AppLoggingService = field(init=False)

    def __post_init__(self) -> None:
        self.runtime_storage = RuntimeStorage(
            config=RuntimeStorageConfig(
                runtime_root=RUNTIME_ROOT,
                data_root=DATA_ROOT,
                keep_last_sessions=10,
            ),
        )
        self.app_logging = AppLoggingService(self.runtime_storage)