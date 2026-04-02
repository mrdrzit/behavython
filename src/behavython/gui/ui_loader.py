from __future__ import annotations

from pathlib import Path
from PySide6 import QtGui, QtUiTools
from behavython.core.app_context import AppContext
from behavython.gui.main_window import BehavythonMainWindow
from behavython.core.paths import ICON_PATH, LOGO_PATH, UI_FILE
from behavython.core.defaults import MAIN_WINDOW_TITLE, LOGO_WIDGET_NAME

def load_ui(ui_file: Path = UI_FILE):
    """
    Load the main .ui file and return the created interface widget.
    """
    loader = QtUiTools.QUiLoader()
    interface = loader.load(str(ui_file))

    if interface is None:
        raise RuntimeError(f"Failed to load UI file: {ui_file}")

    return interface


def apply_branding(interface) -> None:
    """
    Apply window title, icon, and logo pixmap if available.
    """
    interface.setWindowTitle(MAIN_WINDOW_TITLE)

    if ICON_PATH.exists():
        interface.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))

    logo_widget = getattr(interface, LOGO_WIDGET_NAME, None)
    if logo_widget is not None and LOGO_PATH.exists():
        pixmap = QtGui.QPixmap(str(LOGO_PATH))
        logo_widget.setPixmap(pixmap)


def load_main_interface():
    """
    Convenience wrapper for loading and branding the main interface.
    """
    interface = load_ui()
    apply_branding(interface)
    return interface


def bootstrap() -> BehavythonMainWindow:
    """
    Assemble the application and return the main window wrapper.
    """
    context = AppContext()
    interface = load_main_interface()
    window = BehavythonMainWindow(interface=interface, context=context)
    window.show()
    return window
