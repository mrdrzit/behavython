from PySide6.QtCore import QObject, Signal


class TaskSignals(QObject):
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)
    log = Signal(str, str)  # target, message
    warning = Signal(str, str)  # title, message
