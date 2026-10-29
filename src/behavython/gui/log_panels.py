from __future__ import annotations

import logging
from collections.abc import Callable
from PySide6.QtCore import QObject, Signal, Slot
from behavython.core.defaults import GUI_LOG_PANELS


class LoggingService:
    """Writes messages into the GUI log text boxes ("panels")."""

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


class _GuiLogBridge(QObject):
    """Lives on the GUI thread, so records logged from worker threads reach the widgets through a queued signal."""

    message = Signal(str, str)  # panel, text

    def __init__(self, deliver: Callable[[str, str], None]) -> None:
        super().__init__()
        self._deliver = deliver
        self.message.connect(self._on_message)

    @Slot(str, str)
    def _on_message(self, panel: str, text: str) -> None:
        self._deliver(panel, text)


class GuiLogHandler(logging.Handler):
    """Shows INFO and higher records of the modules in GUI_LOG_PANELS in their GUI text box."""

    def __init__(self, panels: LoggingService) -> None:
        super().__init__(level=logging.INFO)
        self._bridge = _GuiLogBridge(panels.append)
        # Longest prefix first, so a specific module wins over its package.
        self._routes = sorted(GUI_LOG_PANELS.items(), key=lambda item: len(item[0]), reverse=True)

    def _panel_for(self, logger_name: str) -> str | None:
        for prefix, panel in self._routes:
            if logger_name == prefix or logger_name.startswith(prefix + "."):
                return panel
        return None

    def emit(self, record: logging.LogRecord) -> None:
        panel = self._panel_for(record.name)
        if panel is None:
            return
        text = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            # logger.exception() keeps the traceback for the file; the panel gets the exception text only.
            text = f"{text}: {record.exc_info[1]}"
        if record.levelno >= logging.WARNING:
            text = f"{record.levelname.capitalize()}: {text}"
        self._bridge.message.emit(panel, text)
