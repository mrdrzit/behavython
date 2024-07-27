import sys
import os
from dlc_helper_functions import *
from PySide6 import QtCore, QtUiTools, QtGui
from PySide6.QtWidgets import QWidget
from PySide6 import QtWidgets, QtCore, QtUiTools, QtGui

class behavython_gui(QWidget):
    """
    This class contains all the commands of the interface as well as the constructors of the
    interface itself.
    """

    def __init__(self):
        """
        This private function calls the interface of a .ui file created in Qt Designer.
        """
        super(behavython_gui, self).__init__()  # Calls the inherited classes __init__ method
        load_gui_path = os.path.join(os.path.dirname(__file__), "behavython_gui.ui")
        loader = QtUiTools.QUiLoader()
        self.interface = loader.load(load_gui_path)  # Loads the interface design archive (made in Qt Designer)
        self.interface.show()
        self.options = {}

        # Create a QThreadPool instance
        self.threadpool = QtCore.QThreadPool()

        # Analysis tab
        self.interface.analysis_button.clicked.connect(lambda: run_analysis(self))
        self.interface.clear_button.clicked.connect(lambda: clear_interface(self))
        self.interface.load_configuration_button.clicked.connect(lambda: load_configuration(self))

        # Deeplabcut tab
        self.interface.get_config_path_button.clicked.connect(lambda: get_folder_path_function(self, "config_path"))
        self.interface.get_videos_path_button.clicked.connect(lambda: get_folder_path_function(self, "videos_path"))
        self.interface.folder_structure_check_button.clicked.connect(lambda: folder_structure_check_function(self))
        self.interface.dlc_video_analyze_button.clicked.connect(lambda: dlc_video_analyze_function(self))
        self.interface.extract_skeleton_button.clicked.connect(lambda: extract_skeleton_function(self))
        self.interface.get_frames_button.clicked.connect(lambda: get_frames_function(self))
        self.interface.clear_unused_files_button.clicked.connect(lambda: clear_unused_files_function(self))

    def resume_message_function(self, file_list):
        text = "Check the videos to be analyzed: "
        message = "The following files are going to be used for pose inference using DeepLabCut:\n\n" + "\n".join(
            file_list + ["\nIs this correct?\n"] + ["If so, click 'Yes' to continue or 'No' to cancel the analysis.\n"]
        )
        answer = option_message_function(self, text, message)
        if answer == "yes":
            return True
        else:
            return False

    def set_icons_and_logo(self):
        logo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "logo", "logo.png"))
        window_icon = os.path.abspath(os.path.join(os.path.dirname(__file__), "logo", "VY.ico"))

        if os.path.exists(window_icon):
            self.interface.setWindowIcon(QtGui.QPixmap(window_icon))
        else:
            print(f"Logo file not found: {logo_path}")

        if os.path.exists(logo_path):
            icon = QtGui.QPixmap(logo_path)
            self.interface.behavython_logo.setPixmap(icon)
        else:
            print(f"Button icon file not found: {logo_path}")

    def run_worker(self, function, *args, **kwargs):
        worker = Worker(function, *args, **kwargs)
        self.threadpool.start(worker)

def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
    window = behavython_gui()  # Create an instance of our class
    os.system("cls")
    startup = get_message()
    os.system(f"echo {startup}")
    app.exec()  # Start the application

if __name__ == "__main__":
    show = main()
