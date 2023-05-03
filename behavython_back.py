import os
import skimage.io
import tkinter as tk
import tkinter.messagebox
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from tkinter import filedialog
from skimage.color import rgb2gray
from scipy import stats
from copy import copy

plt.ioff()


class experiment_class:
    """
    This class declares each experiment and makes the necessary calculations to
    analyze the movement of the animal in the experimental box.

    For each experiment one object of this class will be created and added in
    a experiments list.
    """

    def __init__(self):
        self.name = []  # Experiment name
        self.data = []  # Experiment loaded data
        self.last_frame = []  # Experiment last behavior video frame
        self.directory = []  # Experiment directory

    def video_analyse(self, options):
        self.experiment_type = options["experiment_type"]
        arena_width = options["arena_width"]  # Arena width set by user
        arena_height = options["arena_height"]  # Arena height set by user
        frames_per_second = options["frames_per_second"]  # Video frames per second set by user
        threshold = options["threshold"]  # Motion threshold set by user in Bonsai
        max_video_height = int(
            options["max_fig_res"][0]
        )  # Maximum video height set by user (height is stored in the first element of the list and is converted to int beacuse it comes as a string)
        max_video_width = int(
            options["max_fig_res"][1]
        )  # Maximum video width set by user (width is stored in the second element of the list and is converted to int beacuse it comes as a string)
        plot_options = options["plot_options"]  # Plot options set by user
        video_height, video_width = self.last_frame.shape  # Gets the video height and width from the video's last frame
        factor_width = arena_width / video_width  # Calculates the width scale factor of the video
        factor_height = arena_height / video_height  # Calculates the height scale factor of the video
        number_of_frames = len(self.data)  # Gets the number of frames

        x_axe = self.data[0]  # Gets the x position
        y_axe = self.data[1]  # Gets the y position
        x_axe_cm = self.data[0] * factor_width  # Puts the x position on scale
        y_axe_cm = self.data[1] * factor_height  # Puts the y position on scale
        d_x_axe_cm = (
            np.append(0, np.diff(self.data[0])) * factor_width
        )  # Calculates the step difference of position in x axis
        d_y_axe_cm = (
            np.append(0, np.diff(self.data[1])) * factor_height
        )  # Calculates the step difference of position in y axis

        displacement_raw = np.sqrt(np.square(d_x_axe_cm) + np.square(d_y_axe_cm))
        displacement = displacement_raw
        displacement[displacement < threshold] = 0

        accumulate_distance = np.cumsum(
            displacement
        )  # Sums all the animal's movements and calculates the accumulated distance traveled
        total_distance = max(accumulate_distance)  # Gets the animal's total distance traveled

        time_vector = np.linspace(0, len(self.data) / frames_per_second, len(self.data))  # Creates a time vector
        np.seterr(
            divide="ignore", invalid="ignore"
        )  # Ignores the division by zero at runtime (division by zero is not an error in this case as the are moments when the animal is not moving)
        velocity = np.divide(
            displacement, np.transpose(np.append(0, np.diff(time_vector)))
        )  # Calculates the first derivative and finds the animal's velocity per time
        mean_velocity = np.nanmean(velocity)  # Calculates the mean velocity from the velocity vector

        aceleration = np.divide(
            np.append(0, np.diff(velocity)), np.append(0, np.diff(time_vector))
        )  # Calculates the second derivative and finds the animal's acceleration per time
        movements = np.sum(displacement > 0)  # Calculates the number of movements made by the animal
        time_moving = np.sum(displacement > 0) * (
            1 / frames_per_second
        )  # Calculates the total time of movements made by the animal
        time_resting = np.sum(displacement == 0) * (
            1 / frames_per_second
        )  # Calculates the total time of the animal without movimentations

        kde_space_coordinates = np.array([np.array(x_axe), np.array(y_axe)])
        kde_instance = stats.gaussian_kde(kde_space_coordinates)
        point_density_function = kde_instance.evaluate(kde_space_coordinates)
        color_limits = np.array(
            [
                (x - np.min(point_density_function)) / (np.max(point_density_function) - np.min(point_density_function))
                for x in point_density_function
            ]
        )

        quadrant_data = np.array(self.data[self.data.columns[2:]])  # Extract the quadrant data from csv file
        colDif = np.abs(
            quadrant_data[:, 0] - np.sum(quadrant_data[:][:, 1:], axis=1)
        )  # Here, the values will be off-by-one because MATLAB starts at 1
        full_entry_indexes = colDif != 1  # Create a logical array where there is a "full entry"
        time_spent = np.delete(quadrant_data, full_entry_indexes, 0)  # True crossings over time (full crossings only)
        quadrant_crossings = abs(np.diff(time_spent, axis=0))
        total_time_in_quadrant = np.sum(np.divide(time_spent, frames_per_second), 0)  # Total time spent in each quadrant
        total_number_of_entries = np.sum(quadrant_crossings > 0, 0)  # Total # of entries in each quadrant

        self.analysis_results = {
            "video_width": video_width,
            "video_height": video_height,
            "number_of_frames": number_of_frames,
            "x_axe": x_axe,
            "y_axe": y_axe,
            "x_axe_cm": x_axe_cm,
            "y_axe_cm": y_axe_cm,
            "d_x_axe_cm": d_x_axe_cm,
            "d_y_axe_cm": d_y_axe_cm,
            "displacement": displacement,
            "accumulate_distance": accumulate_distance,
            "total_distance": total_distance,
            "time_vector": time_vector,
            "velocity": velocity,
            "mean_velocity": mean_velocity,
            "aceleration": aceleration,
            "movements": movements,
            "time_spent": time_spent,
            "time_moving": time_moving,
            "time_resting": time_resting,
            "quadrant_crossings": quadrant_crossings,
            "time_in_quadrant": total_time_in_quadrant,
            "number_of_entries": total_number_of_entries,
            "color_limits": color_limits,
            "max_video_height": max_video_height,
            "max_video_width": max_video_width,
            "plot_options": plot_options,
        }

        # print(self.analysis_results)
        if self.experiment_type == "plus_maze":  # If the maze is a plus maze
            dict_to_excel = {
                "Total distance (cm)": total_distance,
                "Mean velocity (cm/s)": mean_velocity,
                "Movements": movements,
                "Time moving (s)": time_moving,
                "Time resting(s)": time_resting,
                "Total time at the upper arm (s)": total_time_in_quadrant[0],
                "Total time at the lower arm (s)": total_time_in_quadrant[1],
                "Total time at the left arm (s)": total_time_in_quadrant[2],
                "Total time at the right arm (s)": total_time_in_quadrant[3],
                "Total time at the center (s)": total_time_in_quadrant[4],
                "Crossings to the upper arm": total_number_of_entries[0],
                "Crossings to the lower arm": total_number_of_entries[1],
                "Crossings to the left arm": total_number_of_entries[2],
                "Crossings to the right arm": total_number_of_entries[3],
                "Crossings to the center": total_number_of_entries[4],
            }
        else:  # If the maze is an open field
            dict_to_excel = {
                "Total distance (cm)": total_distance,
                "Mean velocity (cm/s)": mean_velocity,
                "Movements": movements,
                "Time moving (s)": time_moving,
                "Time resting(s)": time_resting,
                "Total time at the center (s)": total_time_in_quadrant[0],
                "Total time at the edge (s)": total_time_in_quadrant[1],
                "Crossings to the center": total_number_of_entries[0],
                "Crossings to the edge": total_number_of_entries[1],
            }

        data_frame = pd.DataFrame(data=dict_to_excel, index=[self.name])
        data_frame = data_frame.T
        return self.analysis_results, data_frame

    def plot_analysis_pluz_maze(self, plot_viewer, plot_number, save_folder):
        # Figure 1 - Overall Activity in the maze
        plot_option = self.analysis_results["plot_options"]
        movement_points = np.array([self.analysis_results["x_axe"], self.analysis_results["y_axe"]]).T.reshape(-1, 1, 2)
        movement_segments = np.concatenate(
            [movement_points[:-1], movement_points[1:]], axis=1
        )  # Creates a 2D array containing the line segments coordinates
        movement_line_collection = LineCollection(
            movement_segments, cmap="CMRmap", linewidth=1.5
        )  # Creates a LineCollection object with custom color map
        movement_line_collection.set_array(
            self.analysis_results["color_limits"]
        )  # Set the line color to the normalized values of "color_limits"
        line_collection_fig_1 = copy(movement_line_collection)
        line_collection_window = copy(movement_line_collection)  # Create a copy of the line collection object

        figure_1, axe_1 = plt.subplots()
        im = plt.imread(self.directory + ".png")
        axe_1.imshow(im, interpolation="bicubic")
        axe_1.add_collection(line_collection_fig_1)
        axe_1.axis("tight")
        axe_1.axis("off")

        image_height = self.analysis_results["video_height"]
        image_width = self.analysis_results["video_width"]
        max_height = self.analysis_results["max_video_height"]  # Maximum height of desired figure
        max_width = self.analysis_results["max_video_width"]  # Maximum width of desired figure
        figure_dpi = 200  # DPI of the figure
        ratio = min(
            max_height / image_width, max_width / image_height
        )  # Calculate the ratio to be used for image resizing without losing the aspect ratio
        new_resolution_in_inches = (
            image_width * ratio / figure_dpi,
            image_height * ratio / figure_dpi,
        )  # Calculate the new resolution in inches based on the dpi set

        figure_1.subplots_adjust(left=0, right=1, bottom=0, top=1)
        figure_1.set_size_inches(new_resolution_in_inches)

        if plot_option == 0:
            plot_viewer.canvas.axes[plot_number % 9].imshow(
                im, interpolation="bicubic"
            )  # Modulo 9 to make sure the plot number is not out of bounds
            plot_viewer.canvas.axes[plot_number % 9].add_collection(line_collection_window)
            plot_number += 1  # Increment the plot number to be used in the next plot (advance in window)
            plot_viewer.canvas.draw_idle()
        else:
            plt.savefig(
                save_folder + "/" + self.name + "_Overall Activity in the maze.png", frameon="false", dpi=figure_dpi
            )
            plot_viewer.canvas.axes[plot_number % 9].imshow(im, interpolation="bicubic")
            plot_viewer.canvas.axes[plot_number % 9].add_collection(line_collection_window)
            plot_number += 1
            plot_viewer.canvas.draw_idle()

        # Figure 3 - Time spent on each arm over time
        figure_3, ((axe_11, axe_12, axe_13), (axe_21, axe_22, axe_23), (axe_31, axe_32, axe_33)) = plt.subplots(3, 3)
        figure_3.delaxes(axe_11)
        figure_3.delaxes(axe_13)
        figure_3.delaxes(axe_31)
        figure_3.delaxes(axe_33)

        axe_12.plot(self.analysis_results["time_spent"][:, 0], color="#2C53A1")
        entries = np.array(self.analysis_results["quadrant_crossings"][:, 0]) == 1
        axe_12.plot(
            self.analysis_results["quadrant_crossings"][:, 0],
            "o",
            ms=2,
            markevery=entries,
            markerfacecolor="#A21F27",
            markeredgecolor="#A21F27",
        )
        axe_12.set_ylim((0, 1.5))
        axe_12.set_title("Upper arm")

        axe_21.plot(self.analysis_results["time_spent"][:, 1], color="#2C53A1")
        entries = np.array(self.analysis_results["quadrant_crossings"][:, 1]) == 1
        axe_21.plot(
            self.analysis_results["quadrant_crossings"][:, 1],
            "o",
            ms=2,
            markevery=entries,
            markerfacecolor="#A21F27",
            markeredgecolor="#A21F27",
        )
        axe_21.set_ylim((0, 1.5))
        axe_21.set_title("Left  arm")

        axe_22.plot(self.analysis_results["time_spent"][:, 2], color="#2C53A1")
        entries = np.array(self.analysis_results["quadrant_crossings"][:, 2]) == 1
        axe_22.plot(
            self.analysis_results["quadrant_crossings"][:, 2],
            "o",
            ms=2,
            markevery=entries,
            markerfacecolor="#A21F27",
            markeredgecolor="#A21F27",
        )
        axe_22.set_ylim((0, 1.5))
        axe_22.set_title("Center")

        axe_23.plot(self.analysis_results["time_spent"][:, 3], color="#2C53A1")
        entries = np.array(self.analysis_results["quadrant_crossings"][:, 3]) == 1
        axe_23.plot(
            self.analysis_results["quadrant_crossings"][:, 3],
            "o",
            ms=2,
            markevery=entries,
            markerfacecolor="#A21F27",
            markeredgecolor="#A21F27",
        )
        axe_23.set_ylim((0, 1.5))
        axe_23.set_title("Right arm")

        axe_32.plot(self.analysis_results["time_spent"][:, 4], color="#2C53A1")
        entries = np.array(self.analysis_results["quadrant_crossings"][:, 4]) == 1
        axe_32.plot(
            self.analysis_results["quadrant_crossings"][:, 4],
            "o",
            ms=2,
            markevery=entries,
            markerfacecolor="#A21F27",
            markeredgecolor="#A21F27",
        )
        axe_32.set_ylim((0, 1.5))
        axe_32.set_title("Lower arm")

        if plot_option == 1:
            plt.subplots_adjust(hspace=0.8, wspace=0.8)
            plt.savefig(
                save_folder + "/" + self.name + "_Time spent on each area over time.png", frameon="false", dpi=600
            )

        plt.close("all")

    def plot_analysis_open_field(self, plot_viewer, plot_number, save_folder):
        # Figure 1 - Overall Activity in the maze
        plot_option = self.analysis_results["plot_options"]
        movement_points = np.array([self.analysis_results["x_axe"], self.analysis_results["y_axe"]]).T.reshape(-1, 1, 2)
        movement_segments = np.concatenate(
            [movement_points[:-1], movement_points[1:]], axis=1
        )  # Creates a 2D array containing the line segments coordinates
        movement_line_collection = LineCollection(
            movement_segments, cmap="CMRmap", linewidth=1.5
        )  # Creates a LineCollection object with custom color map
        movement_line_collection.set_array(
            self.analysis_results["color_limits"]
        )  # Set the line color to the normalized values of "color_limits"
        line_collection_fig_1 = copy(movement_line_collection)
        line_collection_window = copy(movement_line_collection)  # Create a copy of the line collection object

        figure_1, axe_1 = plt.subplots()
        im = plt.imread(self.directory + ".png")
        axe_1.imshow(im, interpolation="bicubic")
        axe_1.add_collection(line_collection_fig_1)  # Add the line collection to the axe
        axe_1.axis("tight")
        axe_1.axis("off")

        image_height = self.analysis_results["video_height"]
        image_width = self.analysis_results["video_width"]
        max_height = self.analysis_results["max_video_height"]  # Maximum height of desired figure
        max_width = self.analysis_results["max_video_width"]  # Maximum width of desired figure
        ratio = min(
            max_height / image_width, max_width / image_height
        )  # Calculate the ratio to be used for image resizing without losing the aspect ratio
        new_resolution_in_inches = (
            image_width * ratio / 200,
            image_height * ratio / 200,
        )  # Calculate the new resolution in inches based on the dpi set

        figure_1.subplots_adjust(left=0, right=1, bottom=0, top=1)
        figure_1.set_size_inches(new_resolution_in_inches)

        if plot_option == 1:
            plot_viewer.canvas.axes[plot_number % 9].imshow(
                im, interpolation="bicubic"
            )  # Modulo 9 to make sure the plot number is not out of bounds
            plot_viewer.canvas.axes[plot_number % 9].add_collection(line_collection_window)
            plot_number += 1  # Increment the plot number to be used in the next plot (advance in window)
            plot_viewer.canvas.draw_idle()
        else:
            plt.savefig(save_folder + "/" + self.name + "_Overall Activity in the maze.png", frameon="false", dpi=200)
            plot_viewer.canvas.axes[plot_number % 9].imshow(im, interpolation="bicubic")
            plot_viewer.canvas.axes[plot_number % 9].add_collection(line_collection_window)
            plot_number += 1
            plot_viewer.canvas.draw_idle()

        # Figure 3 - Time spent on each area over time
        figure_3, (axe_31, axe_32) = plt.subplots(1, 2)

        axe_31.plot(self.analysis_results["time_spent"][:, 0], color="#2C53A1")
        entries = np.array(self.analysis_results["quadrant_crossings"][:, 0]) == 1
        axe_31.plot(
            self.analysis_results["quadrant_crossings"][:, 0],
            "o",
            ms=2,
            markevery=entries,
            markerfacecolor="#A21F27",
            markeredgecolor="#A21F27",
        )
        axe_31.set_ylim((0, 1.5))
        axe_31.set_title("center")

        axe_32.plot(self.analysis_results["time_spent"][:, 1], color="#2C53A1")
        entries = np.array(self.analysis_results["quadrant_crossings"][:, 1]) == 1
        axe_32.plot(
            self.analysis_results["quadrant_crossings"][:, 1],
            "o",
            ms=2,
            markevery=entries,
            markerfacecolor="#A21F27",
            markeredgecolor="#A21F27",
        )
        axe_32.set_ylim((0, 1.5))
        axe_32.set_title("edge")

        if plot_option == 1:
            plt.subplots_adjust(hspace=0.8, wspace=0.8)
            plt.savefig(
                save_folder + "/" + self.name + "_Time spent on each area over time.png", frameon="false", dpi=600
            )

        # Figure 4 - Number of crossings
        figure_4, (axe_41, axe_42) = plt.subplots(1, 2)

        axe_41.plot(self.analysis_results["quadrant_crossings"][:, 0])
        axe_41.set_ylim((0, 1.5))
        axe_41.set_title("center")

        axe_42.plot(self.analysis_results["quadrant_crossings"][:, 1])
        axe_42.set_ylim((0, 1.5))
        axe_42.set_title("edge")

        if plot_option == 1:
            plt.subplots_adjust(hspace=0.8, wspace=0.8)
            plt.savefig(save_folder + "/" + self.name + "_Number of crossings.png", frameon="false", dpi=600)

        plt.close("all")


class files_class:
    def __init__(self):
        self.number = 0
        self.directory = []
        self.name = []

    def add_files(self, selected_files):
        for file in selected_files:
            name = os.path.basename(file)[:-4]
            if name not in self.name:
                self.name.append(name)
                self.number = self.number + 1
                self.directory.append(file[:-4])


class interface_functions:
    def get_experiments(self, line_edit, experiment_type, save_plots, algo_type="bonsai"):
        if algo_type == "bonsai":
            inexistent_file = 0
            selected_folder_to_save = 0
            error = 0
            file_explorer = tk.Tk()
            file_explorer.withdraw()
            file_explorer.call("wm", "attributes", ".", "-topmost", True)
            selected_files = filedialog.askopenfilename(title="Select the files to analyze", multiple=True)
            if save_plots == 1:
                selected_folder_to_save = filedialog.askdirectory(
                    title="Select the folder to save the plots", mustexist=True
                )
            experiments = []

            try:
                assert selected_folder_to_save != ""
                assert len(selected_files) > 0
            except AssertionError:
                if len(selected_files) == 0:
                    line_edit.append(" ERROR: No files selected")
                    error = 1
                else:
                    line_edit.append(" ERROR: No destination folder selected")
                    error = 2
                return experiments, selected_folder_to_save, error, inexistent_file

            files = files_class()
            files.add_files(selected_files)

            for index in range(0, files.number):
                experiments.append(experiment_class())

                try:
                    raw_data = pd.read_csv(
                        files.directory[index] + ".csv", sep=",", na_values=["no info", "."], header=None
                    )
                    raw_image = rgb2gray(skimage.io.imread(files.directory[index] + ".png"))
                except:
                    line_edit.append("WARNING!! Doesn't exist a CSV or PNG file with the name " + files.name[index])
                    experiments.pop()
                    error = 3
                    inexistent_file = files.name[index]
                    return experiments, selected_folder_to_save, error, inexistent_file
                else:
                    if raw_data.shape[1] == 7 and experiment_type == "plus_maze":
                        experiments[index].data = raw_data.interpolate(
                            method="spline", order=1, limit_direction="both", axis=0
                        )
                        line_edit.append("- File " + files.name[index] + ".csv was read")
                        experiments[index].last_frame = raw_image
                        line_edit.append("- File " + files.name[index] + ".png was read")
                        experiments[index].name = files.name[index]
                        experiments[index].directory = files.directory[index]
                    elif experiment_type == "open_field":
                        experiments[index].data = raw_data.interpolate(
                            method="spline", order=1, limit_direction="both", axis=0
                        )
                        line_edit.append("- File " + files.name[index] + ".csv was read")
                        experiments[index].last_frame = raw_image
                        line_edit.append("- File " + files.name[index] + ".png was read")
                        experiments[index].name = files.name[index]
                        experiments[index].directory = files.directory[index]
                    else:
                        line_edit.append(
                            "WARNING!! The "
                            + files.name[index]
                            + ".csv file had more columns than the elevated plus maze test allows"
                        )
        elif algo_type == "deeplabcut":
            data = DataFiles()
            message = "\nBe careful to select only the files that are relevant to the analysis.\n\nThat being:\n - Skeleton file (csv)\n - Filtered data file (csv)\n - Experiment image (png)\n - Roi file for the area that the mice is supposed to investigate"
            title = "The correct files to select when opening the data to analyze"
            # warning_message_function(title, message)
            inexistent_file = 0
            selected_folder_to_save = 0
            error = 0
            experiments = []
            selected_files = get_files(line_edit, data, experiments)
            if save_plots == 1:
                selected_folder_to_save = filedialog.askdirectory(
                    title="Select the folder to save the plots", mustexist=True
                )

            try:
                assert selected_folder_to_save != ""
                assert len(experiments) > 0
            except AssertionError:
                if len(selected_files) == 0:
                    line_edit.append(" ERROR: No files selected")
                    error = 1
                else:
                    line_edit.append(" ERROR: No destination folder selected")
                    error = 2
                return experiments, selected_folder_to_save, error, inexistent_file
        return experiments, selected_folder_to_save, error, inexistent_file
