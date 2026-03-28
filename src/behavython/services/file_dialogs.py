from __future__ import annotations

from PySide6.QtWidgets import QFileDialog, QWidget


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