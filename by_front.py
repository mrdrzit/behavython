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
        self.debug_mode = False
        self.deeplabcut_is_enabled = False
        self.analysis_failed = False

        # Create a QThreadPool instance
        self.threadpool = QtCore.QThreadPool()

        # Analysis tab
        self.interface.analysis_button.clicked.connect(lambda: self.run_worker(run_analysis, self))
        self.interface.clear_button.clicked.connect(lambda: clear_interface(self))
        self.interface.load_configuration_button.clicked.connect(lambda: self.load_configuration())

        # Deeplabcut tab
        self.interface.get_config_path_button.clicked.connect(lambda: get_folder_path_function(self, "config_path"))
        self.interface.config_path_lineedit.textChanged.connect(lambda: self.enable_DLC())
        self.interface.get_videos_path_button.clicked.connect(lambda: get_folder_path_function(self, "videos_path"))
        self.interface.folder_structure_check_button.clicked.connect(lambda: folder_structure_check_function(self))
        self.interface.dlc_video_analyze_button.clicked.connect(lambda: self.run_worker(dlc_video_analyze_function, self))
        self.interface.extract_skeleton_button.clicked.connect(lambda: self.run_worker(extract_skeleton_function, self))
        self.interface.get_frames_button.clicked.connect(lambda: self.run_worker(get_frames_function, self))
        self.interface.clear_unused_files_button.clicked.connect(lambda: clear_unused_files_function(self))
        self.interface.get_file_to_analyze_button.clicked.connect(lambda: get_folder_path_function(self, "file_to_analyze"))
        self.interface.analyze_from_file_lineedit.textChanged.connect(lambda: self.toggle_analyze_from_file_button())
        self.interface.video_folder_lineedit.textChanged.connect(lambda: self.enable_analysis_buttons())
        self.interface.config_path_lineedit.textChanged.connect(lambda: self.enable_analysis())
        self.interface.analyze_from_file_button.clicked.connect(lambda: self.run_worker(dlc_video_analyze_function, self))
        self.interface.analyze_from_file_extract_skeleton_button.clicked.connect(lambda: self.run_worker(extract_skeleton_function, self))
        self.interface.analyze_from_file_get_frames_button.clicked.connect(lambda: self.run_worker(get_frames_function, self))
        self.interface.create_annotated_video_button.clicked.connect(lambda: self.run_worker(create_annotated_video, self))
        self.interface.config_path_lineedit.textChanged.connect(lambda: self.enable_annotated_video_creation())
        self.interface.folder_to_create_annotated_video_button.clicked.connect(lambda: get_folder_path_function(self, "get_annotated_video_folder"))

        # Data process tab
        self.interface.get_config_path_data_process_button.clicked.connect(lambda: get_folder_path_function(self, "config_path_data_process"))
        self.interface.get_frames_path_data_process_button.clicked.connect(lambda: get_folder_path_function(self, "videos_path_data_process"))
        self.interface.generated_frames_source_button.clicked.connect(lambda: get_folder_path_function(self, "generated_frames_source_data_process_lineedit"))
        self.interface.generated_frames_destination_button.clicked.connect(lambda: get_folder_path_function(self, "generated_frames_destination_data_process_lineedit"))
        self.interface.generated_frames_network_destination_button.clicked.connect(lambda: get_folder_path_function(self, "generated_frames_network_destination_data_process_lineedit"))
        self.interface.generated_frames_merge_button.clicked.connect(lambda: merge_generated_frames(self))
        self.interface.get_source_folder_path_button_video_editing_button.clicked.connect(lambda: get_folder_path_function(self, "source_folder"))
        self.interface.convert_csv_to_h5_data_process_button.clicked.connect(lambda: self.run_worker(convert_csv_to_h5, self))
        self.interface.analyze_frames_data_process_button.clicked.connect(lambda: self.run_worker(analyze_folder_with_frames, self))
        self.interface.enable_bout_analysis_checkbox.stateChanged.connect(lambda: self.enable_bout_analysis())
        self.interface.config_path_data_process_lineedit.textChanged.connect(lambda: self.enable_bout_analysis())
        self.interface.config_path_data_process_lineedit.textChanged.connect(lambda: self.enable_DLC())
        self.interface.run_bout_analysis_button.clicked.connect(lambda: self.run_worker(bout_analysis, self))
        self.interface.get_bout_analysis_folder_button.clicked.connect(lambda: get_folder_path_function(self, "get_bout_analysis_folder"))

        # Video editing tab
        self.interface.get_destination_folder_path_button_video_editing_button.clicked.connect(lambda: get_folder_path_function(self, "destination_folder"))
        self.interface.get_videos_path_video_editing_button.clicked.connect(lambda: get_folder_path_function(self, "crop_path_video_editing"))
        self.interface.get_video_coordinates_video_editing_button.clicked.connect(lambda: self.run_worker(get_crop_coordinates, self))
        self.interface.save_cropped_dimensions_video_editing_button.clicked.connect(lambda: save_crop_coordinates(self))
        self.interface.crop_videos_button_video_editing_button.clicked.connect(lambda: self.run_worker(crop_videos, self))
        self.interface.copy_files_with_robocopy_video_editing_button.clicked.connect(lambda: self.run_worker(copy_folder_robocopy, self))
        self.interface.folder_to_get_create_roi_button.clicked.connect(lambda: get_folder_path_function(self, "create_roi_automatically"))
        self.interface.folder_to_get_create_roi_lineedit.textChanged.connect(lambda: self.enable_DLC())
        self.interface.cread_roi_automatically_button.clicked.connect(lambda: self.run_worker(create_rois_automatically, self))
        self.interface.enable_debugging_mode_checkbox.stateChanged.connect(lambda: self.toggle_debug_mode())
        self.interface.create_validation_video_button.clicked.connect(lambda: create_validation_video(self))

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
        worker.signals.progress.connect(self.update_progress_bar)
        worker.signals.result.connect(handle_results)
        worker.signals.error.connect(lambda error_info: handle_error(self, error_info))
        worker.signals.finished.connect(lambda: on_worker_finished(self))
        worker.signals.data_ready.connect(on_data_ready)
        self.threadpool.start(worker)
    
    def get_files(self, worker, file_type):
        if file_type == "dlc_files":
            worker.stored_data = []
            files, _= QFileDialog.getOpenFileNames(self, "Select the analysis files", "", "DLC files (*.csv *.jpg)")
            worker.stored_data.append(files)
            worker.signals.data_ready.emit()
        elif file_type == "dlc_config":
            worker.stored_data = []
            files, _= QFileDialog.getOpenFileNames(self, "Select the config file", "", "DLC files (*.yaml)")
            worker.stored_data.append(files)
            worker.signals.data_ready.emit()
        elif file_type == "save_folder":
            worker.stored_data = []
            folder = QFileDialog.getExistingDirectory(self, "Select the folder to save the results")
            worker.stored_data.append(folder)
            worker.signals.data_ready.emit()
        elif file_type == "get_options":
            worker.stored_data = []
            options = self.get_options()
            self.options = options
            worker.stored_data.append(options)
            worker.signals.data_ready.emit()
        else:
            print("Bad file request. Please check the file_type argument.")
            worker.signals.data_ready.emit()
            return
    
    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

    def update_lineedit(self, values):
        """Handle multiple line edits with clear/append functionality
        
        Args:
            values (tuple): Can be either:
                - ("clear_all_lineedits") to clear all line edits
                - ("lineedit_name", "clear_lineedit") to clear a specific line edit
                - ("lineedit_name", text) to append text to a specific line edit
        """
        if not values:
            print("No values provided to update_lineedit.")
            return
        
        if len(values) != 2:
            print("Exactly two values required for update_lineedit.")
            return
            
        first, second = values
        
        lineedit_map = {
            "resume_lineedit": self.interface.resume_lineedit,
            "clear_unused_files_lineedit": self.interface.clear_unused_files_lineedit,
            "log_video_editing_lineedit": self.interface.log_video_editing_lineedit,
            "log_data_process_lineedit": self.interface.log_data_process_lineedit,
        }
        
        # Handle clear all case
        if first == "clear_all_lineedits":
            for lineedit in lineedit_map.values():
                if lineedit is not None:
                    lineedit.clear()
            return
        
        # Get the lineedit
        lineedit = lineedit_map.get(first)
        if lineedit is None:
            print(f"Lineedit '{first}' not found.")
            return
        
        # Handle clear or append
        if second == "clear_lineedit":
            lineedit.clear()
        else:
            lineedit.append(second)

    def enable_annotated_video_creation(self):
        elements_to_toggle = [
            self.interface.folder_to_create_annotated_video_button,
            self.interface.folder_to_create_annotated_video_lineedit,
            self.interface.create_annotated_video_button
        ]

        for element in elements_to_toggle:
            element.setEnabled(True if self.interface.config_path_lineedit.text().endswith(".yaml") else False)
        
    def enable_bout_analysis(self):
        analysis_check_box_is_enabled = self.interface.enable_bout_analysis_checkbox.isChecked()
        self.interface.log_data_process_lineedit.clear()

        elements_to_toggle = [
            self.interface.get_bout_analysis_folder_button,
            self.interface.run_bout_analysis_button,
            self.interface.path_to_bout_analysis_folder_lineedit
        ]

        for element in elements_to_toggle:
            element.setEnabled(True if analysis_check_box_is_enabled and self.interface.config_path_data_process_lineedit.text().endswith(".yaml") else False)

        if analysis_check_box_is_enabled:
            if not self.interface.config_path_data_process_lineedit.text().endswith(".yaml"):
                self.interface.log_data_process_lineedit.clear()
                self.interface.log_data_process_lineedit.append("Please select a config file to run the bout analysis.")

    def enable_analysis_buttons(self):
        elements_to_toggle = [
            self.interface.extract_skeleton_button,
            self.interface.get_frames_button,
            self.interface.clear_unused_files_button,
        ]

        for element in elements_to_toggle:
            element.setEnabled(True if not self.interface.video_folder_lineedit.text() == "." else False)

    def enable_analysis(self):
        elements_to_toggle = [
            self.interface.dlc_video_analyze_button,
            self.interface.folder_structure_check_button
        ]

        for element in elements_to_toggle:
            element.setEnabled(True if self.interface.config_path_lineedit.text().endswith(".yaml") else False)

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
            self.interface.analyze_from_file_extract_skeleton_button.setEnabled(False)
            self.interface.analyze_from_file_get_frames_button.setEnabled(False)
        elif not os.path.exists(self.interface.analyze_from_file_lineedit.text()):
            self.interface.resume_lineedit.clear()
            self.interface.resume_lineedit.setText("The file selected does not exist.")
            self.interface.analyze_from_file_button.setEnabled(False)
            self.interface.analyze_from_file_extract_skeleton_button.setEnabled(False)
            self.interface.analyze_from_file_get_frames_button.setEnabled(False)
        elif not self.interface.analyze_from_file_lineedit.text().endswith(".txt"):
            self.interface.resume_lineedit.clear()
            self.interface.resume_lineedit.setText("The file selected is not a .txt file.")
            self.interface.analyze_from_file_button.setEnabled(False)
            self.interface.analyze_from_file_extract_skeleton_button.setEnabled(False)
            self.interface.analyze_from_file_get_frames_button.setEnabled(False)
        else:
            self.interface.analyze_from_file_button.setEnabled(True)
            self.interface.analyze_from_file_extract_skeleton_button.setEnabled(True)
            self.interface.analyze_from_file_get_frames_button.setEnabled(True)
            self.interface.resume_lineedit.clear()

    def get_options(self):
        options = {}
        options["arena_width"] = int(self.interface.arena_width_lineedit.text())
        options["arena_height"] = int(self.interface.arena_height_lineedit.text())
        options["frames_per_second"] = int(self.interface.frames_per_second_lineedit.text())
        options["experiment_type"] = self.interface.type_combobox.currentText().lower().strip().replace(" ", "_")
        options["max_fig_res"] = str(self.interface.fig_max_size.currentText()).replace(" ", "").replace("x", ",").split(",")
        options["algo_type"] = self.interface.algo_type_combobox.currentText().lower().strip()
        if self.interface.animal_combobox.currentIndex() == 0:
            options["threshold"] = 0.0267
        else:
            options["threshold"] = 0.0667
        options["task_duration"] = int(self.interface.task_duration_lineedit.text())
        options["trim_amount"] = int(self.interface.crop_video_time_lineedit.text())
        options["crop_video"] = self.interface.crop_video_checkbox.isChecked()
        options["save_folder"] = ""
        options["plot_options"] = "plotting_enabled" if self.interface.plot_data_checkbox.isChecked() else "plotting_disabled"

        return options
    
    def toggle_debug_mode(self):
        """Toggle debug mode and visual indicators"""
        debug_mode = self.interface.enable_debugging_mode_checkbox.isChecked()
        
        # Change window color based on debug state
        if debug_mode:
            self.debug_mode = True
            
            # Apply orange background
            self.interface.setStyleSheet("""
                QWidget {
                    background-color: #598d96;
                    color: #FFFFFF;
                }
            """)
            
            self.update_lineedit(("clear_all_lineedits", ""))
            self.update_lineedit(("resume_lineedit", "🐞 Debug mode active"))
            self.update_lineedit(("clear_unused_files_lineedit", "🐞 Debug mode active"))
            self.update_lineedit(("log_video_editing_lineedit", "🐞 Debug mode active"))
            self.update_lineedit(("log_data_process_lineedit", "🐞 Debug mode active"))
        else:
            # Revert to normal styling
            self.interface.setStyleSheet("""
                QWidget {
                    background-color: #4B4B4B;
                    color: #FFFFFF;
                }
            """)
            self.update_lineedit(("clear_all_lineedits", ""))
            self.update_lineedit(("resume_lineedit", "🔧 Debug mode deactivated"))
            self.update_lineedit(("clear_unused_files_lineedit", "🔧 Debug mode deactivated"))
            self.update_lineedit(("log_video_editing_lineedit", "🔧 Debug mode deactivated"))
            self.update_lineedit(("log_data_process_lineedit", "🔧 Debug mode deactivated"))
            self.debug_mode = False

    def enable_DLC(self):
        lineedits_to_check = [
            self.interface.config_path_lineedit,
            self.interface.config_path_data_process_lineedit,
            self.interface.folder_to_get_create_roi_lineedit
        ]

        if any(lineedit.text() != "" for lineedit in lineedits_to_check):
            self.deeplabcut_is_enabled = True
        else:
            self.deeplabcut_is_enabled = False

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
