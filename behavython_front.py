import behavython_back
import sys
import os
import json
from dlc_helper_functions import *
from behavython_plot_widget import plot_viewer
from PySide6 import QtWidgets, QtCore, QtUiTools
from PySide6.QtWidgets import QMainWindow, QMessageBox
from PySide6.QtCore import QObject, QThread, Signal


class analysis_class(QObject):
    """This class is a wrapper for the analysis function. It is used to run the analysis after the data is loaded."""

    finished = Signal()  # Signal that will be output to the interface when the function is completed
    progress_bar = Signal(int)

    def __init__(self, experiments, options, plot_viewer):
        """
        This private function is executed when the class is called, and all parameters are
        defined here
        """
        super(analysis_class, self).__init__()
        self.experiments = experiments
        self.options = options
        self.plot_viewer = plot_viewer

    def run_analyse(self, options):
        for i in range(0, len(self.experiments)):
            if options["algo_type"] == "deeplabcut":
                analyse_results, data_frame = behavython_back.experiment_class.video_analyse(self, self.options, self.experiments[i])
                pass
            else:
                analyse_results, data_frame = self.experiments[i].video_analyse(self.options)
            if i == 0:
                results_data_frame = data_frame
            else:
                results_data_frame = results_data_frame.join(data_frame)

            if self.options["experiment_type"] == "open_field":
                self.experiments[i].plot_analysis_open_field(self.plot_viewer, i, self.options["save_folder"])
            elif self.options["experiment_type"] == "plus_maze":
                self.experiments[i].plot_analysis_pluz_maze(self.plot_viewer, i, self.options["save_folder"])
            elif self.options["experiment_type"] == "njr" or self.options["experiment_type"] == "social_recognition":
                behavython_back.experiment_class.plot_analysis_social_behavior(
                    self, self.plot_viewer, i, self.options["save_folder"]
                )
            self.progress_bar.emit(round(((i + 1) / len(self.experiments)) * 100))

        if self.options["plot_options"] in "plotting_enabled":
            results_data_frame.T.to_excel(self.options["save_folder"] + "/analysis_results.xlsx")
        config_file_path = self.options["save_folder"] + "/analysis_configuration.json"
        config_json = options_to_configuration(self.options)
        with open(config_file_path, "w") as config_file:
            json.dump(config_json, config_file, indent=4, sort_keys=True)
        self.finished.emit()


class behavython_gui(QMainWindow):
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
        os.environ["PYSIDE_DESIGNER_PLUGINS"] = os.path.join(os.path.dirname(__file__))
        loader = QtUiTools.QUiLoader()
        loader.registerCustomWidget(plot_viewer)
        self.interface = loader.load(load_gui_path)  # Loads the interface design archive (made in Qt Designer)
        self.interface.show()
        self.options = {}
        self.interface.clear_button.clicked.connect(self.clear_function)
        self.interface.analysis_button.clicked.connect(self.analysis_function)

        ## This block handles the deeplabcut analysis
        self.interface.folder_structure_check_button.clicked.connect(lambda: folder_structure_check_function(self))
        self.interface.dlc_video_analyze_button.clicked.connect(lambda: dlc_video_analyze_function(self))
        self.interface.extract_skeleton_button.clicked.connect(lambda: extract_skeleton_function(self))
        self.interface.clear_unused_files_button.clicked.connect(lambda: clear_unused_files_function(self))
        self.interface.get_config_path_button.clicked.connect(lambda: get_folder_path_function(self, "config_path"))
        self.interface.get_videos_path_button.clicked.connect(lambda: get_folder_path_function(self, "videos_path"))
        self.interface.get_frames_button.clicked.connect(lambda: get_frames_function(self))
        self.interface.load_configuration_button.clicked.connect(lambda: self.load_configuration(self))

    def analysis_function(self):
        self.interface.resume_lineedit.clear()
        self.options["arena_width"] = int(self.interface.arena_width_lineedit.text())
        self.options["arena_height"] = int(self.interface.arena_height_lineedit.text())
        self.options["frames_per_second"] = int(self.interface.frames_per_second_lineedit.text())
        self.options["experiment_type"] = self.interface.type_combobox.currentText().lower().strip().replace(" ", "_")
        self.options["plot_options"] = "plotting_enabled" if self.interface.save_button.isChecked() else "plotting_disabled"
        # Remove trailing spaces and replace x with comma and split the values at the comma to make a list
        self.options["max_fig_res"] = str(self.interface.fig_max_size.currentText()).replace(" ", "").replace("x", ",").split(",")
        self.options["algo_type"] = self.interface.algo_type_combobox.currentText().lower().strip()
        if self.interface.animal_combobox.currentIndex() == 0:
            self.options["threshold"] = 0.0267
        else:
            self.options["threshold"] = 0.0667
        self.options["task_duration"] = int(self.interface.task_duration_lineedit.text())
        self.options["trim_amount"] = int(self.interface.crop_video_time_lineedit.text())
        self.options["crop_video"] = self.interface.crop_video_checkbox.isChecked()

        functions = behavython_back.interface_functions()
        if self.options["algo_type"] == "deeplabcut":
            [self.experiments, save_folder, error_flag, inexistent_file] = functions.get_experiments(
                self.interface.resume_lineedit,
                self.options["experiment_type"],
                self.options["plot_options"],
                self.options["algo_type"],
            )
        else:
            [self.experiments, save_folder, error_flag, inexistent_file] = functions.get_experiments(
                self.interface.resume_lineedit,
                self.options["experiment_type"],
                self.options["plot_options"],
                self.options["algo_type"],
            )
        if error_flag != 0:
            if error_flag == 1:
                warning_message_function(
                    "File selection problem", "WARNING!! No files were selected. Please select a file and try again."
                )
            elif error_flag == 2:
                warning_message_function(
                    "File selection problem",
                    "WARNING!! No destination folder was selected. Please select a valid folder and try again.",
                )
            elif error_flag == 3:
                warning_message_function(
                    "File selection problem", "WARNING!! Doesn't exist a CSV or PNG file with the name " + inexistent_file
                )
            sys.exit(4)
        self.options["save_folder"] = save_folder

        # Creates a QThread object to plot the received data
        self.analysis_thread = QThread()
        # Creates a worker object named plot_data_class
        self.analysis_worker = analysis_class(self.experiments, self.options, self.interface.plot_viewer)
        # Moves the class to the thread
        self.analysis_worker.moveToThread(self.analysis_thread)
        # When the process is finished, this command quits the worker
        self.analysis_worker.finished.connect(self.analysis_thread.quit)
        # When the process is finished, this command waits the worker to finish completely
        self.analysis_worker.finished.connect(self.analysis_thread.wait)
        # When the process is finished, this command deletes the worker
        self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)
        # Updates the progress bar
        self.analysis_worker.progress_bar.connect(self.progress_bar_function)
        # When the process is finished, this command deletes the thread.
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)
        # Starts the thread
        self.analysis_thread.start()
        # Runs the analysis to be executed in the thread
        self.analysis_worker.run_analyse(self.options)

    def progress_bar_function(self, value):
        self.interface.progress_bar.setValue(value)

    def clear_function(self, configuration={}):
        if configuration:
            # self.options = configuration
            if configuration["Experiment Type"].lower().strip().replace(" ", "_") == "njr":
                self.interface.type_combobox.setCurrentIndex(0)
            elif configuration["Experiment Type"].lower().strip().replace(" ", "_") == "social_recognition":
                self.interface.type_combobox.setCurrentIndex(1)
            elif configuration["Experiment Type"].lower().strip().replace(" ", "_") == "plus_maze":
                self.interface.type_combobox.setCurrentIndex(2)
            elif configuration["Experiment Type"].lower().strip().replace(" ", "_") == "open_field":
                self.interface.type_combobox.setCurrentIndex(3)
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid experiment type.")
                self.clear_function
                return
            
            if configuration["Algorithm Type"].lower().strip().replace(" ", "_") == "deeplabcut":
                self.interface.algo_type_combobox.setCurrentIndex(0)
            elif configuration["Algorithm Type"].lower().strip().replace(" ", "_") == "bonsai":
                self.interface.algo_type_combobox.setCurrentIndex(1)
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid algorithm type.")
                self.clear_function
                return
            
            if configuration["Arena width"] > 0 and configuration["Arena width"] < 650:
                self.interface.arena_width_lineedit.setText(str(int(configuration["Arena width"])))
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid arena width.")
                self.clear_function
                return
            
            if configuration["Arena height"] > 0 and configuration["Arena height"] < 650:
                self.interface.arena_height_lineedit.setText(str(int(configuration["Arena height"])))
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid arena height.")
                self.clear_function
                return
            
            if configuration["Video framerate"] > 0 and configuration["Video framerate"] < 240:
                self.interface.frames_per_second_lineedit.setText(str(int(configuration["Video framerate"])))
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid number of frames per second.")
                self.clear_function
                return
            
            if configuration["Experimental Animal"].lower().strip().replace(" ", "_") == "mouse":
                self.interface.animal_combobox.setCurrentIndex(0)
            elif configuration["Experimental Animal"].lower().strip().replace(" ", "_") == "rat":
                self.interface.animal_combobox.setCurrentIndex(1)
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid experimental animal.\n We support mice and rats at the time.")
                self.clear_function
                return

            if configuration["Task Duration"] > 0 and configuration["Task Duration"] < 86400:
                self.interface.task_duration_lineedit.setText(str(int(configuration["Task Duration"])))
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid task duration. We support up to 24 hours of task duration.")
                self.clear_function
                return

            if configuration["Amount to trim"] > 0 and configuration["Amount to trim"] < configuration["Task Duration"]:
                self.interface.crop_video_time_lineedit.setText(str(int(configuration["Amount to trim"])))
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid trim amount. You cannot trim more than the video duration.")
                self.clear_function
                return

            resolutions = [['640', '480'], ['1280', '720'], ['1920', '1080'], ['2560', '1440']]
            if configuration["Plot resolution"] in resolutions:
                self.interface.fig_max_size.setCurrentIndex(resolutions.index(configuration["Plot resolution"]))
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid figure resolution.")
                self.clear_function
                return

            if configuration["Plot option"] == "plotting_enabled":
                self.interface.save_button.setChecked(True)
                self.interface.only_plot_button.setChecked(False)
            elif configuration["Plot option"] == "plotting_disabled":
                self.interface.save_button.setChecked(False)
                self.interface.only_plot_button.setChecked(True)
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid plot option.")
                self.clear_function
                return

            if configuration["Crop video"] == True:
                self.interface.crop_video_checkbox.setChecked(True)
            elif configuration["Crop video"] == False:
                self.interface.crop_video_checkbox.setChecked(False)
            else:
                warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid crop video option.")
                self.clear_function
                return
            
            self.interface.clear_unused_files_lineedit.clear()
            self.interface.resume_lineedit.clear()
            self.clear_plot()
        else:
            self.options = {}
            self.interface.arena_width_lineedit.setText("30")
            self.interface.arena_height_lineedit.setText("30")
            self.interface.type_combobox.setCurrentIndex(0)
            self.interface.frames_per_second_lineedit.setText("30")
            self.interface.animal_combobox.setCurrentIndex(0)
            self.interface.algo_type_combobox.setCurrentIndex(0)
            self.interface.clear_unused_files_lineedit.clear()
            self.interface.resume_lineedit.clear()
            self.interface.task_duration_lineedit.setText("300")
            self.interface.crop_video_time_lineedit.setText("15")
            self.interface.crop_video_checkbox.setChecked(True)
            self.interface.fig_max_size.setCurrentIndex(1)
            self.interface.only_plot_button.setChecked(False)
            self.interface.save_button.setChecked(True)
            self.clear_plot()

    def clear_plot(self):
        for i in range(1, 10):
            self.interface.plot_viewer.canvas.axes[i - 1].cla()  # Changes the plot face color

    def resume_message_function(self, file_list):
        text = "Check the videos to be analyzed: "
        message = "The following files are going to be used for pose inference using DeepLabCut:\n\n" + "\n".join(
            file_list + ["\nIs this correct?\n"] + ["If so, click 'Yes' to continue or 'No' to cancel the analysis.\n"]
        )
        answer = self.option_message_function(text, message)
        if answer == "yes":
            return True
        else:
            return False
    def load_configuration(self, config_path):
        config_path = file_selection_function(self)
        if not test_configuration_file(config_path):
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.")
            return
        else:
            configuration = json.load(open(config_path))
            self.clear_function(configuration)
            self.interface.resume_lineedit.setText("Configuration file loaded successfully!")

    def option_message_function(self, text, info_text):
        dialog = CustomDialog(text, info_text, self.interface)
        result = dialog.exec()

        if result == QDialog.Accepted:
            return "yes"
        else:
            return "no"

def warning_message_function(title, text):
    warning = QMessageBox()  # Create the message box
    warning.setWindowTitle(title)  # Message box title
    warning.setText(text)  # Message box text
    warning.setIcon(QMessageBox.Icon.Warning)  # Message box icon
    warning.setStyleSheet(
        "QMessageBox{background:#353535;}QLabel{font:10pt/DejaVu Sans/;"
        + "font-weight:bold;color:#FFFFFF;}QPushButton{width:52px; border:2px solid #A21F27;border-radius:8px;"
        + "background-color:#2C53A1;color:#FFFFFF;font:10pt/DejaVu Sans/;"
        + "font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;"
        + "border-radius:8px;background-color:#A21F27;color:#FFFFFF;}"
    )
    warning.setStandardButtons(QMessageBox.StandardButton.Ok)  # Message box buttons
    warning.exec()

def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)  # Create an instance of QtWidgets.QApplication
    window = behavython_gui()  # Create an instance of our class

    # -----------------------------------------------------------------------------------------
    ## This block of code is to be used when the interface is going to be compiled into an exe
    ## It serves the function of removing the splash when the program finishes loading
    # try:
    #     import pyi_splash

    #     pyi_splash.close()
    # except:
    #     pass
    # -----------------------------------------------------------------------------------------
    os.system("cls")
    os.system("echo Behavython loaded successfully!")
    app.exec()  # Start the application


if __name__ == "__main__":
    show = main()
