from __future__ import annotations

from dataclasses import dataclass, field
from PySide6.QtCore import QThreadPool
from src.behavython.core.paths import DATA_ROOT, RUNTIME_ROOT
from src.behavython.pipeline.models import RuntimeStorageConfig
from src.behavython.services.logging import AppLoggingService, register_logging_service
from src.behavython.services.storage import RuntimeStorage

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
        register_logging_service(self.app_logging)