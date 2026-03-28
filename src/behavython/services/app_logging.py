from __future__ import annotations

import io
import logging
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from logging.handlers import RotatingFileHandler
from pathlib import Path

from behavython.services.runtime_storage import RuntimeStorage


class _LineBuffer(io.TextIOBase):
    def __init__(self, logger: logging.Logger, level: int) -> None:
        super().__init__()
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._buffer += text

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.strip()
            if line:
                self.logger.log(self.level, line)

        return len(text)

    def flush(self) -> None:
        remaining = self._buffer.strip()
        if remaining:
            self.logger.log(self.level, remaining)
        self._buffer = ""


class AppLoggingService:
    def __init__(self, runtime_storage: RuntimeStorage) -> None:
        self.runtime_storage = runtime_storage

        self.app_logger = logging.getLogger("behavython")
        self.dlc_logger = logging.getLogger("behavython.dlc")
        self.external_logger = logging.getLogger("behavython.external")

        self._configure()

    def _configure(self) -> None:
        self.app_logger.setLevel(logging.DEBUG)
        self.dlc_logger.setLevel(logging.DEBUG)
        self.external_logger.setLevel(logging.DEBUG)

        self.app_logger.propagate = False
        self.dlc_logger.propagate = False
        self.external_logger.propagate = False

        self._clear_handlers(self.app_logger)
        self._clear_handlers(self.dlc_logger)
        self._clear_handlers(self.external_logger)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        persistent_log_path = self.runtime_storage.logs_root / "app.log"
        session_log_path = self.runtime_storage.session_logs_dir / "session.log"
        external_log_path = self.runtime_storage.session_dlc_output_dir / "external_output.log"

        persistent_handler = RotatingFileHandler(
            persistent_log_path,
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        persistent_handler.setLevel(logging.DEBUG)
        persistent_handler.setFormatter(formatter)

        session_handler = logging.FileHandler(session_log_path, encoding="utf-8")
        session_handler.setLevel(logging.DEBUG)
        session_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)

        external_handler = logging.FileHandler(external_log_path, encoding="utf-8")
        external_handler.setLevel(logging.DEBUG)
        external_handler.setFormatter(formatter)

        self.app_logger.addHandler(persistent_handler)
        self.app_logger.addHandler(session_handler)
        self.app_logger.addHandler(console_handler)

        self.dlc_logger.addHandler(persistent_handler)
        self.dlc_logger.addHandler(session_handler)
        self.dlc_logger.addHandler(console_handler)

        self.external_logger.addHandler(external_handler)
        self.external_logger.addHandler(session_handler)

        self._quiet_noisy_loggers()

    def _clear_handlers(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _quiet_noisy_loggers(self) -> None:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)

    @contextmanager
    def capture_external_output(self, logger_name: str = "behavython.external"):
        logger = logging.getLogger(logger_name)
        stdout_buffer = _LineBuffer(logger, logging.INFO)
        stderr_buffer = _LineBuffer(logger, logging.ERROR)

        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            try:
                yield
            finally:
                stdout_buffer.flush()
                stderr_buffer.flush()   