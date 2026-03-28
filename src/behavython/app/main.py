from __future__ import annotations

import os
import sys

from PySide6 import QtCore, QtWidgets

from behavython.app.bootstrap import bootstrap
from behavython.dlc_helper_functions import get_message


def main() -> int:
    """
    Application entry point.
    """
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    app = QtWidgets.QApplication(sys.argv)
    _window = bootstrap()

    try:
        os.system("cls")
        startup = get_message()
        os.system(f"echo {startup}")
    except Exception:
        pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())