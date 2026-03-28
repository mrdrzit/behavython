from __future__ import annotations

import io
import logging
import warnings
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from logging.handlers import RotatingFileHandler
from behavython.services.runtime_storage import RuntimeStorage


class _FilteredExternalStream(io.TextIOBase):
    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        passthrough_stream,
        allow_terminal_progress: bool = True,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.level = level
        self.passthrough_stream = passthrough_stream
        self.allow_terminal_progress = allow_terminal_progress
        self._buffer = ""

    def _is_tqdm_text(self, text: str) -> bool:
        tqdm_markers = (
            "%|",
            "it/s]",
            "|█",
        )
        return any(marker in text for marker in tqdm_markers)

    def _should_skip_line(self, line: str) -> bool:
        noisy_patterns = (
            'Call to CreateProcess failed. Error code: 2, command: \'"ptxas.exe"',
            "Couldn't get ptxas version string",
            "Relying on driver to perform ptx compilation.",
            "Modify $PATH to customize ptxas location.",
            "This message will be only logged once.",
            "TensorFloat-32 will be used for the matrix multiplication.",
        )
        return any(pattern in line for pattern in noisy_patterns)

    def _handle_text_fragment(self, text: str) -> None:
        if not text:
            return

        if self.allow_terminal_progress and self._is_tqdm_text(text):
            self.passthrough_stream.write(text)
            self.passthrough_stream.flush()

    def _emit_line(self, line: str) -> None:
        stripped_line = line.strip()
        if not stripped_line:
            return

        if self._is_tqdm_text(stripped_line):
            return

        if self._should_skip_line(stripped_line):
            return

        self.logger.log(self.level, stripped_line)

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._handle_text_fragment(text)
        self._buffer += text

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)

        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
        self._buffer = ""

        try:
            self.passthrough_stream.flush()
        except Exception:
            pass


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

        self.console_logger = logging.getLogger("behavython.console")
        self.console_logger.setLevel(logging.INFO)
        self.console_logger.propagate = False

        self._clear_handlers(self.console_logger)

        console_only_handler = logging.StreamHandler()
        console_only_handler.setLevel(logging.INFO)
        console_only_handler.setFormatter(logging.Formatter("%(message)s"))

        self.console_logger.addHandler(console_only_handler)

        self._quiet_noisy_loggers()

    def _clear_handlers(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _quiet_noisy_loggers(self) -> None:
        logging.captureWarnings(True)

        logging.getLogger("py.warnings").setLevel(logging.WARNING)
        logging.getLogger("deeplabcut").setLevel(logging.WARNING)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("h5py").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.WARNING)

        warnings.filterwarnings(
            "ignore",
            message="Starting a Matplotlib GUI outside of the main thread will likely fail.",
        )

    @contextmanager
    def capture_external_output(self, logger_name: str = "behavython.external"):
        logger = logging.getLogger(logger_name)

        stdout_stream = _FilteredExternalStream(
            logger=logger,
            level=logging.INFO,
            passthrough_stream=sys.__stdout__,
            allow_terminal_progress=True,
        )
        stderr_stream = _FilteredExternalStream(
            logger=logger,
            level=logging.ERROR,
            passthrough_stream=sys.__stderr__,
            allow_terminal_progress=True,
        )

        with redirect_stdout(stdout_stream), redirect_stderr(stderr_stream):
            try:
                yield
            finally:
                stdout_stream.flush()
                stderr_stream.flush()


_active_logging_service: AppLoggingService | None = None


def register_logging_service(service: AppLoggingService) -> None:
    global _active_logging_service
    _active_logging_service = service


@contextmanager
def capture_external_output(logger_name: str = "behavython.external"):
    if _active_logging_service is None:
        yield
        return

    with _active_logging_service.capture_external_output(logger_name=logger_name):
        yield
