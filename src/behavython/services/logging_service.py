from __future__ import annotations


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
