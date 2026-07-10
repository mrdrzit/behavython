from __future__ import annotations

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QPlainTextEdit, QVBoxLayout, QWidget
from PySide6.QtWidgets import QFileDialog, QDialogButtonBox

def show_warning(parent: QWidget | None, title: str, text: str) -> None:
    QMessageBox.warning(parent, title, text)


def show_info(parent: QWidget | None, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


class ScrollableQuestionDialog(QDialog):
    def __init__(self, parent: QWidget | None, title: str, text: str) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(400, 300)
        self.resize(600, 400)
        self.setModal(True)

        layout = QVBoxLayout(self)

        self.text_edit = QPlainTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(text)
        self.text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.text_edit, 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)


def ask_yes_no(parent: QWidget | None, title: str, text: str) -> bool:
    dialog = ScrollableQuestionDialog(parent, title, text)
    # PySide6 exec() can return int (1 for Accepted) or QDialog.DialogCode.Accepted
    # Depending on the PySide6 version, casting or comparing to int is safest 
    # to avoid Enum equality issues, or simply compare to QDialog.DialogCode.Accepted
    return int(dialog.exec()) == int(QDialog.DialogCode.Accepted)

def show_worker_error(parent: QWidget | None, error_info: tuple) -> None:
    dialog = WorkerErrorDialog(parent, error_info)
    dialog.exec()

def select_file(parent: QWidget, title: str, filter_str: str = "All Files (*)") -> str:
    path, _ = QFileDialog.getOpenFileName(parent, title, "", filter_str)
    return path


def select_files(parent: QWidget, title: str, filter_str: str = "All Files (*)") -> list[str]:
    paths, _ = QFileDialog.getOpenFileNames(parent, title, "", filter_str)
    return paths


def select_folder(parent: QWidget, title: str) -> str:
    return QFileDialog.getExistingDirectory(parent, title)


def select_save_folder(parent: QWidget, title: str = "Select output folder") -> str:
    return QFileDialog.getExistingDirectory(parent, title)

class WorkerErrorDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None,
        error_info: tuple,
        title: str = "Worker error",
    ) -> None:
        super().__init__(parent)

        exctype, value, traceback_text = error_info
        exception_name = getattr(exctype, "__name__", "Exception")

        self.setWindowTitle(title)
        self.setMinimumSize(820, 520)
        self.resize(980, 700)
        self.setModal(True)

        layout = QVBoxLayout(self)

        summary_label = QLabel(f"<b>{exception_name}:</b> {value}")
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        hint_label = QLabel("The task failed. The full traceback is shown below.")
        hint_label.setWordWrap(True)
        layout.addWidget(hint_label)

        self.traceback_edit = QPlainTextEdit()
        self.traceback_edit.setReadOnly(True)
        self.traceback_edit.setPlainText(traceback_text)
        self.traceback_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.traceback_edit, 1)

        button_row = QHBoxLayout()
        button_row.addStretch()

        self.copy_button = QPushButton("Copy traceback")
        self.copy_button.clicked.connect(self.copy_traceback)
        button_row.addWidget(self.copy_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_row.addWidget(self.close_button)

        layout.addLayout(button_row)

    def copy_traceback(self) -> None:
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.traceback_edit.toPlainText())
