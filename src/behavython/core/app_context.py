from __future__ import annotations

from dataclasses import dataclass, field
from PySide6.QtCore import QThreadPool
from behavython.services.logging import AppLoggingService, register_logging_service


@dataclass
class AppContext:
    threadpool: QThreadPool = field(default_factory=QThreadPool)
    debug_mode: bool = False
    app_logging: AppLoggingService = field(init=False)

    def __post_init__(self) -> None:
        self.app_logging = AppLoggingService()
        register_logging_service(self.app_logging)
