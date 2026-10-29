from __future__ import annotations

import io
import logging
import warnings
import os
import sys
import re
from contextlib import contextmanager, redirect_stderr, redirect_stdout, suppress
from datetime import datetime
from pathlib import Path
from behavython.core.defaults import SESSION_LOGS_TO_KEEP
from behavython.core.paths import USER_LOGS_ROOT

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
                        logger.info(msg)
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


class _ShortNameFormatter(logging.Formatter):
    """Exposes the logger name without the shared "behavython." prefix as %(short_name)s."""

    def format(self, record: logging.LogRecord) -> str:
        record.short_name = record.name.removeprefix("behavython.")
        return super().format(record)


def create_session_log_path(logs_root: Path = USER_LOGS_ROOT, keep: int = SESSION_LOGS_TO_KEEP) -> Path:
    """Returns a new timestamped log file path and deletes the oldest session logs beyond `keep`."""
    logs_root.mkdir(parents=True, exist_ok=True)

    existing = sorted(logs_root.glob("*.log"))
    for old_log in existing[: max(len(existing) - (keep - 1), 0)]:
        with suppress(OSError):  # still open by another running session
            old_log.unlink()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = logs_root / f"{stamp}.log"
    counter = 1
    while path.exists():
        counter += 1
        path = logs_root / f"{stamp}_{counter}.log"
    return path


class AppLoggingService:
    """Configures the "behavython" logger tree: one file per session plus the terminal.

    Modules log with logging.getLogger(__name__); those loggers have no handlers of their own and propagate here.
    """

    def __init__(self, is_cli: bool = False, console_level: int = logging.INFO) -> None:
        self.is_cli = is_cli
        self.console_level = console_level
        self.session_log_path = create_session_log_path()
        self._configure()

    def _configure(self) -> None:
        root = logging.getLogger("behavython")
        root.setLevel(logging.DEBUG)
        root.propagate = False
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()

        file_formatter = _ShortNameFormatter(fmt="%(asctime)s | %(levelname)-7s | %(short_name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        if self.is_cli:
            console_formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
        else:
            console_formatter = file_formatter

        session_handler = logging.FileHandler(self.session_log_path, encoding="utf-8")
        session_handler.setLevel(logging.DEBUG)
        session_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(console_formatter)
        if not self.is_cli:
            # The GUI terminal only shows behavython messages; DLC and TensorFlow output stays in the file.
            console_handler.addFilter(lambda record: not record.name.startswith("behavython.external"))

        root.addHandler(session_handler)
        root.addHandler(console_handler)

        self._quiet_noisy_loggers()

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
