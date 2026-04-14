from __future__ import annotations

import traceback
import logging
from PySide6.QtCore import QRunnable, Slot
from behavython.tasks.signals import TaskSignals


class TaskWorker(QRunnable):
    def __init__(self, fn, request, debug_mode: bool = False):
        super().__init__()
        self.fn = fn
        self.request = request
        self.debug_mode = debug_mode
        self.signals = TaskSignals()
        self.logger = logging.getLogger("behavython")

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("Worker started: %s", getattr(self.fn, "__name__", str(self.fn)))

            if self.debug_mode:
                try:
                    import debugpy

                    if debugpy.is_client_connected():
                        debugpy.debug_this_thread()
                except Exception:
                    self.logger.exception("Failed to attach debugpy to worker thread.")

            result = self.fn(
                self.request,
                progress=self.signals.progress,
                log=self.signals.log,
                warning=self.signals.warning,
            )
        except Exception as exc:
            self.logger.exception("Worker failed: %s", getattr(self.fn, "__name__", str(self.fn)))
            self.signals.error.emit((type(exc), str(exc), traceback.format_exc()))
        else:
            self.logger.info("Worker finished successfully: %s", getattr(self.fn, "__name__", str(self.fn)))
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
