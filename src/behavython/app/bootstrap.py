from __future__ import annotations

from behavython.app.app_context import AppContext
from behavython.gui.main_window import BehavythonMainWindow
from behavython.gui.ui_loader import load_main_interface


def bootstrap() -> BehavythonMainWindow:
    """
    Assemble the application and return the main window wrapper.
    """
    context = AppContext()
    interface = load_main_interface()
    window = BehavythonMainWindow(interface=interface, context=context)
    window.show()
    return window
