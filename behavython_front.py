import behavython_back
import sys
import os
import tkinter as tk
import deeplabcut
from pathlib import Path
from tkinter import filedialog
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.QtCore import QObject, QThread, pyqtSignal


class analysis_class(QObject):
    """This class is a wrapper for the analysis function. It is used to run the analysis after the data is loaded."""

    finished = pyqtSignal()  # Signal that will be output to the interface when the function is completed
    progress_bar = pyqtSignal(int)

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
                analyse_results, data_frame = behavython_back.experiment_class.video_analyse(
                    self, self.options, self.experiments[i]
                )
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
            elif self.options["experiment_type"] == "social_behavior":
                behavython_back.experiment_class.plot_analysis_social_behavior(
                    self, self.plot_viewer, i, self.options["save_folder"]
                )
            self.progress_bar.emit(round(((i + 1) / len(self.experiments)) * 100))

        if self.options["plot_options"] == 1:
            results_data_frame.T.to_excel(self.options["save_folder"] + "/analysis_results.xlsx")
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
        uic.loadUi(load_gui_path, self)  # Loads the interface design archive (made in Qt Designer)
        self.show()
        self.options = {}
        self.clear_button.clicked.connect(self.clear_function)
        self.analysis_button.clicked.connect(self.analysis_function)

        ## This block handles the deeplabcut analysis
        self.folder_structure_check_button.clicked.connect(self.folder_structure_check_function)
        self.dlc_video_analyze_button.clicked.connect(self.dlc_video_analyze_function)
        self.get_data_files_button.clicked.connect(self.get_data_files_function)
        self.extract_skeleton_button.clicked.connect(self.extract_skeleton_function)
        self.clear_unused_files_button.clicked.connect(self.clear_unused_files_function)
        self.get_config_path_button.clicked.connect(lambda: self.get_folder_path_function("config_path"))
        self.get_videos_path_button.clicked.connect(lambda: self.get_folder_path_function("videos_path"))

    def analysis_function(self):
        self.resume_lineedit.clear()
        self.options["arena_width"] = int(self.arena_width_lineedit.text())
        self.options["arena_height"] = int(self.arena_height_lineedit.text())
        self.options["frames_per_second"] = float(self.frames_per_second_lineedit.text())
        self.options["experiment_type"] = self.type_combobox.currentText().lower().strip().replace(" ", "_")
        self.options["plot_options"] = self.save_button.isChecked()
        # Remove trailing spaces and replace x with comma and split the values at the comma to make a list
        self.options["max_fig_res"] = str(self.fig_max_size.currentText()).replace(" ", "").replace("x", ",").split(",")
        self.options["algo_type"] = self.algo_type_combobox.currentText().lower().strip()
        if self.animal_combobox.currentIndex() == 0:
            self.options["threshold"] = 0.0267
        else:
            self.options["threshold"] = 0.0667
        self.options["task_duration"] = int(self.crop_video_lineedit.text())
        self.options["trim_amount"] = int(self.crop_video_time_lineedit.text())
        self.options["crop_video"] = self.crop_video_checkbox.isChecked()

        functions = behavython_back.interface_functions()
        if self.options["algo_type"] == "deeplabcut":
            [self.experiments, save_folder, error_flag, inexistent_file] = functions.get_experiments(
                self.resume_lineedit,
                self.options["experiment_type"],
                self.options["plot_options"],
                self.options["algo_type"],
            )
        else:
            [self.experiments, save_folder, error_flag, inexistent_file] = functions.get_experiments(
                self.resume_lineedit,
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
        self.analysis_worker = analysis_class(self.experiments, self.options, self.plot_viewer)
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
        self.progress_bar.setValue(value)

    def clear_function(self):
        self.options = {}
        self.type_combobox.setCurrentIndex(0)
        self.frames_per_second_lineedit.setText("30")
        self.arena_width_lineedit.setText("65")
        self.arena_height_lineedit.setText("65")
        self.animal_combobox.setCurrentIndex(0)
        self.fig_max_size.setCurrentIndex(0)
        self.algo_type_combobox.setCurrentIndex(0)
        self.clear_plot()

    def clear_plot(self):
        for i in range(1, 10):
            self.plot_viewer.canvas.axes[i - 1].cla()  # Changes the plot face color

    def folder_structure_check_function(self):
        self.clear_unused_files_lineedit.clear()
        folder_path = os.path.dirname(self.config_path_lineedit.text().replace('"', "").replace("'", ""))
        if folder_path == "":
            message = "Please, select a path to the config.yaml file before checking the folder structure."
            title = "Path to config file not selected"
            warning_message_function(title, message)

        required_folders = ["dlc-models", "evaluation-results", "labeled-data", "training-datasets", "videos"]
        required_files = ["config.yaml"]

        for folder in required_folders:
            if not os.path.isdir(os.path.join(folder_path, folder)):
                self.clear_unused_files_lineedit.append(f"The folder '{folder}' is NOT present")
                return False
            self.clear_unused_files_lineedit.append(f"The folder {folder} is OK")

        for file in required_files:
            if not os.path.isfile(os.path.join(folder_path, file)):
                self.clear_unused_files_lineedit.append(f"The project's {file} is NOT present")
                return False
        # Check if dlc-models contains at least one iteration folder
        dlc_models_path = os.path.join(folder_path, "dlc-models")
        iteration_folders = [
            f
            for f in os.listdir(dlc_models_path)
            if os.path.isdir(os.path.join(dlc_models_path, f)) and f.startswith("iteration-")
        ]
        if not iteration_folders:
            self.clear_unused_files_lineedit.append("There are no iteration folders in dlc-models.")
            return False

        latest_iteration_folder = max(iteration_folders, key=lambda x: int(x.split("-")[1]))
        shuffle_set = os.listdir(os.path.join(dlc_models_path, latest_iteration_folder))
        if not shuffle_set:
            self.clear_unused_files_lineedit.append("There are no shuffle sets in the latest iteration folder.")
            return False
        else:
            for root, dirs, files in os.walk(os.path.join(dlc_models_path, latest_iteration_folder, shuffle_set[0])):
                for dir in dirs:
                    if dir.startswith("log"):
                        continue
                    if "train" not in dirs or "test" not in dirs:
                        self.clear_unused_files_lineedit.append("The train or test folder is missing.")
                        return False
                    if dir.startswith("test") and not os.path.isfile(os.path.join(root, dir, "pose_cfg.yaml")):
                        self.clear_unused_files_lineedit.append("The pose_cfg.yaml file is missing in test folder.")
                        return False
                    if dir.startswith("train"):
                        if not os.path.isfile(os.path.join(root, dir, "pose_cfg.yaml")):
                            self.clear_unused_files_lineedit.append("The pose_cfg.yaml file is missing in test folder.")
                            return False
                        elif not any("meta" in string for string in os.listdir(os.path.join(root, dir))):
                            self.clear_unused_files_lineedit.append("The meta file is missing in train folder.")
                            return False
                        elif not any("data" in string for string in os.listdir(os.path.join(root, dir))):
                            self.clear_unused_files_lineedit.append("The data file is missing in train folder.")
                            return False
                        elif not any("index" in string for string in os.listdir(os.path.join(root, dir))):
                            self.clear_unused_files_lineedit.append("The index file is missing in train folder.")
                            return False

        # If all checks pass, the folder structure is correct
        self.clear_unused_files_lineedit.append("The folder structure is correct.")
        return True

    def dlc_video_analyze_function(self):
        self.clear_unused_files_lineedit.clear()
        self.clear_unused_files_lineedit.append(f"Using DeepLabCut version{deeplabcut.__version__}")
        config_path = self.config_path_lineedit.text().replace('"', "").replace("'", "")
        videos = self.video_folder_lineedit.text().replace('"', "").replace("'", "")
        _, _, file_list = [entry for entry in os.walk(videos)][0]
        file_extension = file_list[0].split(".")[-1]

        all_has_same_extension = all([file.split(".")[-1] == file_extension for file in file_list])
        if not all_has_same_extension:
            title = "Video extension error"
            message = "All videos must have the same extension.\n Please, check the videos folder and try again."
            warning_message_function(title, message)
            return False

        self.clear_unused_files_lineedit.append("Analyzing videos...")
        deeplabcut.analyze_videos(
            config_path,
            videos,
            videotype=file_extension,
            shuffle=1,
            trainingsetindex=0,
            gputouse=0,
            allow_growth=True,
            save_as_csv=True,
        )
        self.clear_unused_files_lineedit.append("Done analyzing videos.")

        self.clear_unused_files_lineedit.append("Filtering data files and saving as CSV...")
        deeplabcut.filterpredictions(
            config_path,
            videos,
            videotype=file_extension,
            shuffle=1,
            trainingsetindex=0,
            filtertype="median",
            windowlength=5,
            p_bound=0.001,
            ARdegree=3,
            MAdegree=1,
            alpha=0.01,
            save_as_csv=True,
        )
        self.clear_unused_files_lineedit.append("Done filtering data files")

    def get_data_files_function(self):
        pass

    def extract_skeleton_function(self):
        self.clear_unused_files_lineedit.clear()
        self.clear_unused_files_lineedit.append(f"Using DeepLabCut version{deeplabcut.__version__}")
        config_path = self.config_path_lineedit.text().replace('"', "").replace("'", "")
        videos = self.video_folder_lineedit.text().replace('"', "").replace("'", "")
        _, _, file_list = [entry for entry in os.walk(videos)][0]
        file_extension = file_list[0].split(".")[-1]

        self.clear_unused_files_lineedit.append("Extracting skeleton...")
        deeplabcut.analyzeskeleton(config_path, videos, shuffle=1, trainingsetindex=0, filtered=True, save_as_csv=True)
        self.clear_unused_files_lineedit.append("Done extracting skeleton.")

    def clear_unused_files_function(self):
        self.clear_unused_files_lineedit.clear()
        config_path = self.config_path_lineedit.text().replace('"', "").replace("'", "")
        videos = self.video_folder_lineedit.text().replace('"', "").replace("'", "")
        _, _, file_list = [entry for entry in os.walk(videos)][0]
        file_extension = file_list[0].split(".")[-1]

        for file in file_list:
            if (
                file.endswith(file_extension)
                or file.endswith(".png")
                or file.endswith(".jpg")
                or file.endswith(".tiff")
                or "roi" in file
            ):
                continue
            if file.endswith(".h5") or file.endswith(".pickle") or "filtered" not in file:
                os.remove(os.path.join(videos, file))
                self.clear_unused_files_lineedit.append(f"Removed {file}")
        _, _, file_list = [entry for entry in os.walk(videos)][0]

        has_filtered_csv = False
        has_skeleton_filtered_csv = False
        has_roi_file = False
        has_left_roi_file = False
        has_right_roi_file = False
        has_image_file = False
        missing_files = []
        task_type = self.type_combobox.currentText().lower().strip().replace(" ", "_")

        for file in file_list:
            if file.endswith("filtered.csv"):
                has_filtered_csv = True
                continue
            elif file.endswith("filtered_skeleton.csv"):
                has_skeleton_filtered_csv = True
                continue
            elif file.endswith(".png") or file.endswith(".jpg") or file.endswith(".tiff"):
                has_image_file = True
                continue
            if task_type == "njr":
                if file.endswith("roiD"):
                    has_right_roi_file = True
                    continue
                elif file.endswith("roiE"):
                    has_left_roi_file = True
                    continue
            elif task_type == "social_recognition":
                if file.endswith("roi.csv"):
                    has_roi_file = True
                    continue
        if any(
            [
                not has_filtered_csv,
                not has_skeleton_filtered_csv,
                not has_left_roi_file,
                not has_right_roi_file,
                not has_roi_file,
                not has_image_file,
            ]
        ):
            self.clear_unused_files_lineedit.append("There are missing files in the folder")
        else:
            self.clear_unused_files_lineedit.append("All required files are present.")
            return
        if not has_filtered_csv:
            missing_files.append(" - filtered.csv")
        if not has_skeleton_filtered_csv:
            missing_files.append(" - skeleton_filtered.csv")
        if not has_image_file:
            missing_files.append(" - screenshot of the video")
        if task_type == "njr":
            if not has_left_roi_file:
                missing_files.append(" - roiD.csv")
            if not has_right_roi_file:
                missing_files.append(" - roiE.csv")
        if task_type == "social_recognition" and not has_roi_file:
            missing_files.append(" - roi.csv")

        title = "Missing files"
        message = "The following files are missing:\n\n" + "\n".join(
            missing_files
            + ["\nPlease, these files are essential for the analysis to work.\nCheck the analysis folder and try again."]
        )
        warning_message_function(title, message)

    def get_folder_path_function(self, lineedit_name):
        if lineedit_name == "config_path":
            file_explorer = tk.Tk()
            file_explorer.withdraw()
            file_explorer.call("wm", "attributes", ".", "-topmost", True)
            config_file = str(Path(filedialog.askopenfilename(title="Select the config.yaml file", multiple=False)))
            self.config_path_lineedit.setText(config_file)
        elif lineedit_name == "videos_path":
            file_explorer = tk.Tk()
            file_explorer.withdraw()
            file_explorer.call("wm", "attributes", ".", "-topmost", True)
            folder = str(Path(filedialog.askdirectory(title="Select the folder", mustexist=True)))
            self.video_folder_lineedit.setText(folder)


def warning_message_function(title, text):
    warning = QMessageBox()  # Create the message box
    warning.setWindowTitle(title)  # Message box title
    warning.setText(text)  # Message box text
    warning.setIcon(QMessageBox.Warning)  # Message box icon
    warning.setStyleSheet(
        "QMessageBox{background:#353535;}QLabel{font:10pt/DejaVu Sans/;"
        + "font-weight:bold;color:#FFFFFF;}QPushButton{width:52px; border:2px solid #A21F27;border-radius:8px;"
        + "background-color:#2C53A1;color:#FFFFFF;font:10pt/DejaVu Sans/;"
        + "font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;"
        + "border-radius:8px;background-color:#A21F27;color:#FFFFFF;}"
    )
    warning.setStandardButtons(QMessageBox.Ok)  # Message box buttons
    warning.exec_()


def main():
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

    app.exec_()  # Start the application


if __name__ == "__main__":
    show = main()
