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
        self.set_icons_and_logo()
        self.interface.show()
        self.options = {}
        self.progress_bar = self.interface.progress_bar

        # Create a QThreadPool instance
        self.threadpool = QtCore.QThreadPool.globalInstance()

        # Analysis tab
        self.interface.analysis_button.clicked.connect(lambda: run_analysis(self))
        self.interface.clear_button.clicked.connect(lambda: clear_interface(self))
        self.interface.load_configuration_button.clicked.connect(lambda: self.load_configuration())

        # Deeplabcut tab
        self.interface.get_config_path_button.clicked.connect(lambda: get_folder_path_function(self, "config_path"))
        self.interface.get_videos_path_button.clicked.connect(lambda: get_folder_path_function(self, "videos_path"))
        self.interface.folder_structure_check_button.clicked.connect(lambda: folder_structure_check_function(self))
        self.interface.dlc_video_analyze_button.clicked.connect(lambda: self.run_worker(dlc_video_analyze_function, self))
        self.interface.extract_skeleton_button.clicked.connect(lambda: self.run_worker(extract_skeleton_function, self))
        self.interface.get_frames_button.clicked.connect(lambda: self.run_worker(get_frames_function, self))
        self.interface.clear_unused_files_button.clicked.connect(lambda: clear_unused_files_function(self))
        self.interface.get_file_to_analyze_button.clicked.connect(lambda: get_folder_path_function(self, "file_to_analyze"))
        self.interface.analyze_from_file_lineedit.textChanged.connect(lambda: self.toggle_analyze_from_file_button())
        self.interface.analyze_from_file_button.clicked.connect(lambda: self.run_worker(dlc_video_analyze_function, self))

        # Data process tab
        self.interface.get_config_path_data_process_button.clicked.connect(lambda: get_folder_path_function(self, "config_path_data_process"))
        self.interface.get_frames_path_data_process_button.clicked.connect(lambda: get_folder_path_function(self, "videos_path_data_process"))
        self.interface.get_videos_path_video_editing_button.clicked.connect(lambda: get_folder_path_function(self, "crop_path_video_editing"))
        self.interface.get_source_folder_path_button_video_editing_button.clicked.connect(lambda: get_folder_path_function(self, "source_folder"))
        self.interface.get_destination_folder_path_button_video_editing_button.clicked.connect(lambda: get_folder_path_function(self, "destination_folder"))
        self.interface.convert_csv_to_h5_data_process_button.clicked.connect(lambda: self.run_worker(convert_csv_to_h5, self))
        self.interface.analyze_frames_data_process_button.clicked.connect(lambda: self.run_worker(analyze_folder_with_frames, self))
        self.interface.enable_bout_analysis_checkbox.stateChanged.connect(lambda: self.enable_bout_analysis())
        self.interface.run_bout_analysis_button.clicked.connect(lambda: self.run_worker(bout_analysis, self))
        self.interface.get_bout_analysis_folder_button.clicked.connect(lambda: get_folder_path_function(self, "get_bout_analysis_folder"))

        # Video editing tab
        self.interface.get_video_coordinates_video_editing_button.clicked.connect(lambda: self.run_worker(get_crop_coordinates, self))
        self.interface.save_cropped_dimensions_video_editing_button.clicked.connect(lambda: save_crop_coordinates(self))
        self.interface.crop_videos_button_video_editing_button.clicked.connect(lambda: self.run_worker(crop_videos, self))
        self.interface.copy_files_with_robocopy_video_editing_button.clicked.connect(lambda: self.run_worker(copy_folder_robocopy, self))
        self.interface.folder_to_get_create_roi_button.clicked.connect(lambda: get_folder_path_function(self, "create_roi_automatically"))
        self.interface.cread_roi_automatically_button.clicked.connect(lambda: self.run_worker(create_rois_automatically, self))

    def resume_message_function(self, file_list):
        text = "Check the videos to be analyzed: "
        message = "The following files are going to be used for pose inference using DeepLabCut:\n\n" + "\n".join(
            file_list + ["\nIs this correct?\n"] + ["If so, click 'Yes' to continue or 'No' to cancel the analysis.\n"]
        )
        # answer = option_message_function(self, text, message)
        answer = "yes"
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
        worker.signals.text_signal.connect(self.update_lineedit)
        worker.signals.warning_message.connect(self.handle_warning_message)
        worker.signals.request_files.connect(lambda file_type: self.get_files(worker, file_type))
        self.threadpool.start(worker)
    
    def get_files(self, worker, file_type):
        # debugpy.debug_this_thread()
        if file_type == "dlc_files":
            files, _= QFileDialog.getOpenFileNames(self, "Select the analysis files", "", "DLC files (*.csv *.jpg)")
            worker.stored_data.append(files)
        elif file_type == "dlc_config":
            files, _= QFileDialog.getOpenFileNames(self, "Select the config file", "", "DLC files (*.yaml)")
            worker.stored_data.append(files)
        else:
            print("Bad file request. Please check the file_type argument.")
            return
        
    
    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

    def update_lineedit(self, values):
        text, lineedit = values
        if "resume_lineedit" == lineedit:
            self.interface.resume_lineedit.append(text)
        elif "clear_unused_files_lineedit" == lineedit:
            if "clear_lineedit" == text:
                self.interface.clear_unused_files_lineedit.clear()
            else:
                self.interface.clear_unused_files_lineedit.append(text)
        elif "log_video_editing_lineedit" == lineedit:
            self.interface.log_video_editing_lineedit.append(text)
        elif "log_data_process_lineedit" == lineedit:
            if "clear_lineedit" == text:
                self.interface.clear_unused_files_lineedit.clear()
            else:
                self.interface.log_data_process_lineedit.append(text)

    def enable_bout_analysis(self):
        analysis_check_box_is_enabled = self.interface.enable_bout_analysis_checkbox.isChecked()
        if analysis_check_box_is_enabled:
            self.interface.get_bout_analysis_folder_button.setEnabled(True)
        else:
            self.interface.get_bout_analysis_folder_button.setEnabled(False)

        if analysis_check_box_is_enabled:
            self.interface.run_bout_analysis_button.setEnabled(True)
        else:
            self.interface.run_bout_analysis_button.setEnabled(False)

        if analysis_check_box_is_enabled:
            self.interface.path_to_bout_analysis_folder_lineedit.setEnabled(True)
        else:
            self.interface.path_to_bout_analysis_folder_lineedit.setEnabled(False)

    def handle_warning_message(self, results):
        title, text = results
        warning_message_function(title, text)

    def handle_resume_message(self, file_list):
        self.resume_message_function(file_list)

    def load_configuration(self, progress_callback=None):
        config_path = file_selection_function(self)
        if not test_configuration_file(config_path):
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.")
            return
        else:
            configuration = json.load(open(config_path))
            load_configuration_file(self, configuration)
            self.interface.resume_lineedit.setText("Configuration file loaded successfully!")
    
    def toggle_analyze_from_file_button(self):
        if self.interface.analyze_from_file_lineedit.text() == "":
            self.interface.resume_lineedit.clear()
            self.interface.resume_lineedit.setText("Please select a file to analyze.")
            self.interface.analyze_from_file_button.setEnabled(False)
        elif not os.path.exists(self.interface.analyze_from_file_lineedit.text()):
            self.interface.resume_lineedit.clear()
            self.interface.resume_lineedit.setText("The file selected does not exist.")
            self.interface.analyze_from_file_button.setEnabled(False)
        elif not self.interface.analyze_from_file_lineedit.text().endswith(".txt"):
            self.interface.resume_lineedit.clear()
            self.interface.resume_lineedit.setText("The file selected is not a .txt file.")
            self.interface.analyze_from_file_button.setEnabled(False)
        else:
            self.interface.analyze_from_file_button.setEnabled(True)
            self.interface.resume_lineedit.clear()

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
