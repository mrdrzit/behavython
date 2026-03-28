from __future__ import annotations

from behavython.workers.task_worker import TaskWorker


class TaskRunner:
    def __init__(self, threadpool, on_log, on_warning, on_progress, on_result, on_error, on_finished):
        self.threadpool = threadpool
        self.on_log = on_log
        self.on_warning = on_warning
        self.on_progress = on_progress
        self.on_result = on_result
        self.on_error = on_error
        self.on_finished = on_finished

    def submit(self, fn, request, debug_mode: bool = False) -> None:
        worker = TaskWorker(fn, request, debug_mode=debug_mode)
        worker.signals.log.connect(self.on_log)
        worker.signals.warning.connect(self.on_warning)
        worker.signals.progress.connect(self.on_progress)
        worker.signals.result.connect(self.on_result)
        worker.signals.error.connect(self.on_error)
        worker.signals.finished.connect(self.on_finished)
        self.threadpool.start(worker)