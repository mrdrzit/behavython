from __future__ import annotations

import io
import logging
import warnings
import os
import sys
import re
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from logging.handlers import RotatingFileHandler
from behavython.core.defaults import LOGGING_NAME_MAP
from behavython.pipeline.models import MappedFormatter
from behavython.services.storage import RuntimeStorage

console = logging.getLogger("behavython.console")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class LoggingService:
    def __init__(self, interface):
        self.interface = interface
        self._map = {
            "resume": self.interface.resume_lineedit,
            "dlc": self.interface.clear_unused_files_lineedit,
        }

    def append(self, target: str, message: str) -> None:
        widget = self._map.get(target)
        if widget is not None:
            widget.append(message)

    def clear(self, target: str) -> None:
        widget = self._map.get(target)
        if widget is not None:
            widget.clear()

    def clear_all(self) -> None:
        for widget in self._map.values():
            widget.clear()


class _FilteredExternalStream(io.TextIOBase):
    def __init__(
        self,
        logger: logging.Logger,
        level: int,
        is_cli: bool = False,
    ) -> None:
        super().__init__()
        self.logger = logger
        self.level = level
        self._buffer = ""
        self._last_percent = -1
        self.is_cli = is_cli

    def _is_tqdm_text(self, text: str) -> bool:
        tqdm_markers = ("%|", "it/s]", "|█")
        return any(marker in text for marker in tqdm_markers)

    def _should_skip_line(self, line: str) -> bool:
        noisy_patterns = (
            # System Noise
            'Call to CreateProcess failed. Error code: 2, command: \'"ptxas.exe"',
            "Couldn't get ptxas version string",
            "Relying on driver to perform ptx compilation.",
            "Modify $PATH to customize ptxas location.",
            "This message will be only logged once.",
            "TensorFloat-32 will be used for the matrix multiplication.",
            "Could not load dynamic library",
            "Cannot dlopen some GPU libraries",
            "Skipping registering GPU devices",
            "cudart64_",
            "cublas64_",
            "cudnn64_",
            "cufft64_",
            "curand64_",
            "cusolver64_",
            "cusparse64_",
            "MLIR V1 optimization pass is not enabled",
            "This TensorFlow binary is optimized with oneAPI",
            "light mode",
            # DLC Fluff
            "The videos are analyzed. Now your research can truly start!",
            "You can create labeled videos with",
            "If the tracking is not satisfactory",
            "extract_outlier_frames",
            "Saving filtered csv poses!",
            "Starting to extract posture",
            "Saving csv poses!",
            "Processing ",
            "Loading ",
        )
        return any(pattern in line for pattern in noisy_patterns)

    def _emit_line(self, line: str) -> None:
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_line = ansi_escape.sub("", line)
        stripped_line = clean_line.strip()

        # tqdm sometimes leaves trailing bracket artifacts if escapes were mangled
        if stripped_line == "[A" or not stripped_line:
            return

        if self._is_tqdm_text(stripped_line):
            # Milestone-based progress (every 10%)
            match = re.search(r"(\d+)%", stripped_line)
            if match:
                percent = int(match.group(1))
                if percent % 10 == 0 and percent != self._last_percent:
                    speed_match = re.search(r"([0-9.]+\s*(?:it/s|s/it))", stripped_line)
                    speed_str = f" | {speed_match.group(1)}" if speed_match else ""
                    msg = f"Progress: {percent}%{speed_str}".replace("it/s", " frames/sec")
                    self.logger.log(self.level, msg)
                    if not self.is_cli:
                        logging.getLogger("behavython.console").info(msg)
                    self._last_percent = percent
            return

        if self._should_skip_line(stripped_line):
            return

        self.logger.log(self.level, stripped_line)

    def write(self, text: str) -> int:
        if not text:
            return 0

        self._buffer += text

        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit_line(line)

        return len(text)

    def flush(self) -> None:
        if self._buffer:
            self._emit_line(self._buffer)
        self._buffer = ""


class CLIQuietFilter(logging.Filter):
    """Silences redundant or interactive-only logs for the CLI."""

    def filter(self, record: logging.LogRecord) -> bool:
        noise = (
            "Calling deeplabcut.",
            "Filtering predictions with DeepLabCut for",
        )
        msg = record.getMessage()
        return not any(p in msg for p in noise)


class AppLoggingService:
    def __init__(
        self,
        runtime_storage: RuntimeStorage,
        is_cli: bool = False,
        console_level: int = logging.INFO,
    ) -> None:
        self.runtime_storage = runtime_storage
        self.is_cli = is_cli
        self.console_level = console_level

        self.app_logger = logging.getLogger("behavython")
        self.dlc_logger = logging.getLogger("behavython.dlc")
        self.external_logger = logging.getLogger("behavython.external")
        self.console_logger = logging.getLogger("behavython.console")
        self.cli_logger = logging.getLogger("behavython.cli")

        self._configure()

    def _configure(self) -> None:
        # Base level for all managed loggers
        for lg in [self.app_logger, self.dlc_logger, self.external_logger, self.console_logger, self.cli_logger]:
            lg.setLevel(logging.DEBUG)
            lg.propagate = False  # Each is a branch head for our logging system
            self._clear_handlers(lg)

        # 1. Formatters
        # Rich formatter for file logs and GUI console
        file_formatter = MappedFormatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)-4s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            name_map=LOGGING_NAME_MAP,
        )

        # Compact formatter for CLI terminal
        if self.is_cli:
            console_formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%H:%M:%S",
            )
        else:
            console_formatter = file_formatter

        # 2. Handlers
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
        persistent_handler.setFormatter(file_formatter)

        session_handler = logging.FileHandler(session_log_path, encoding="utf-8")
        session_handler.setLevel(logging.DEBUG)
        session_handler.setFormatter(file_formatter)

        # Unified Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(console_formatter)

        if self.is_cli:
            console_handler.addFilter(CLIQuietFilter())

        external_handler = logging.FileHandler(external_log_path, encoding="utf-8")
        external_handler.setLevel(logging.DEBUG)
        external_handler.setFormatter(file_formatter)

        # 3. Attachment
        # Core loggers get file + terminal
        for lg in [self.app_logger, self.dlc_logger, self.cli_logger]:
            lg.addHandler(persistent_handler)
            lg.addHandler(session_handler)
            lg.addHandler(console_handler)

        # Console logger only gets terminal (used for specific SYSTEM messages)
        self.console_logger.addHandler(console_handler)

        # External logger (TF/DLC output) always gets file logs
        self.external_logger.addHandler(external_handler)
        self.external_logger.addHandler(session_handler)

        # In CLI mode, we also want external output in the terminal
        if self.is_cli:
            self.external_logger.addHandler(console_handler)

        self._quiet_noisy_loggers()

    def _clear_handlers(self, logger: logging.Logger) -> None:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def _quiet_noisy_loggers(self) -> None:
        """Normally silences external loggers, but unlocked for auditing."""
        logging.captureWarnings(True)

        logging.getLogger("py.warnings").setLevel(logging.ERROR)
        logging.getLogger("deeplabcut").setLevel(logging.DEBUG)
        logging.getLogger("matplotlib").setLevel(logging.ERROR)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("torch").setLevel(logging.DEBUG)
        logging.getLogger("h5py").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)

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
            is_cli=self.is_cli,
            # passthrough_stream=sys.__stdout__,
            # allow_terminal_progress=True,
        )
        stderr_stream = _FilteredExternalStream(
            logger=logger,
            level=logging.INFO,
            is_cli=self.is_cli,
            # passthrough_stream=sys.__stdout__,
            # allow_terminal_progress=True,
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
