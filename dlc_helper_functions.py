import math
import os
import re
import subprocess
import json
import base64
import random
import traceback
import shutil
import cv2
import csv
import re
import matplotlib
import threading
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import tkinter as tk
import itertools as it
import seaborn as sns
import matplotlib.cm as cm
from tkinter import filedialog
from tqdm import tqdm
from PIL import Image
from scipy import stats
from pathlib import Path
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from PySide6.QtWidgets import QMessageBox, QFileDialog, QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget
from PySide6.QtGui import QFontMetrics
from PySide6.QtCore import QRunnable, Slot, Signal, QObject, QThread

import debugpy

matplotlib.use("agg")
DLC_ENABLE = True
if DLC_ENABLE:
    import deeplabcut

data_ready_event = threading.Event()
def on_data_ready():
    data_ready_event.set()

class CustomDialog(QDialog):
    def __init__(self, text, info_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warning")

        layout = QVBoxLayout()

        # Scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        # Widget to hold the content
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)

        label = QLabel(text)
        info_label = QLabel(info_text)

        content_layout.addWidget(label)
        content_layout.addWidget(info_label)
        scroll_area.setWidget(content_widget)

        # Apply stylesheet to the scroll area and its viewport
        scroll_area.setStyleSheet("QScrollArea {background-color:#353535; border:none;}")
        scroll_area.viewport().setStyleSheet("background-color:#353535; border:none;")

        layout.addWidget(scroll_area)

        # Buttons
        self.button_yes = QPushButton("    YES    ")
        self.button_no = QPushButton("     NO      ")

        self.button_yes.clicked.connect(self.accept)
        self.button_no.clicked.connect(self.reject)

        layout.addWidget(self.button_yes)
        layout.addWidget(self.button_no)

        self.setLayout(layout)

        # StyleSheet
        self.setStyleSheet(
            "QDialog{background-color:#353535;}"  # Set dialog background
            "QLabel{font:10pt 'DejaVu Sans'; font-weight:bold; color:#FFFFFF;}"  # Set label font and color
            "QPushButton{width:52px; border:2px solid #A21F27; border-radius:8px; background-color:#2C53A1; color:#FFFFFF; font:10pt 'DejaVu Sans'; font-weight:bold;}"  # Set button style
            "QPushButton:pressed{border:2px solid #A21F27; border-radius:8px; background-color:#A21F27; color:#FFFFFF;}"  # Set button pressed style
        )

        # Calculate the required width based on the longest line of text
        font_metrics = QFontMetrics(info_label.font())
        longest_line_width = max(font_metrics.horizontalAdvance(line) for line in info_text.split("\n"))
        # Add some padding to the width
        padding = 200
        self.setFixedWidth(longest_line_width + padding)

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

class experiment_class:
    """
    This class declares each experiment and makes the necessary  calculations to
    analyze the movement of the animal in the experimental box.

    For each experiment one object of this class will be created and added in
    a experiments list.
    """

    def __init__(self):
        self.name = []  # Experiment name
        self.data = []  # Experiment loaded data
        self.last_frame = []  # Experiment last behavior video frame
        self.directory = []  # Experiment directory

class WorkerSignals(QObject):
    """
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    """

    finished = Signal()  # QtCore.Signal
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)
    finished = Signal()
    text_signal = Signal(tuple)
    warning_message = Signal(object)
    resume_message = Signal(object)
    request_files = Signal(str)
    data_ready = Signal()

class Worker(QRunnable):
    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    parameters:
        fn: The function to run on this worker thread.
        args: The arguments to pass to the function fn.
        kwargs: The keyword arguments to pass to the function fn.

    """

    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.stored_data = []
        self.kwargs["progress"] = self.signals.progress
        self.kwargs["text_signal"] = self.signals.text_signal
        self.kwargs["warning_message"] = self.signals.warning_message
        self.kwargs["resume_message"] = self.signals.resume_message
        self.kwargs["request_files"] = self.signals.request_files

    @Slot()
    def run(self):
        try:
            result = self.function(self, *self.args, **self.kwargs)
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit((type(e), str(e), traceback.format_exc()))
        else:
            self.signals.result.emit((result, self.args[0]))
        finally:
            self.signals.finished.emit()

class video_editing_tool:
    """
    A class that represents a video editing tool.

    Attributes:
        image_list (list): A list of image paths.
        image_names (list): A list of image names.
        modified_images (set): A set of image names that have been modified.
        image_dictionary (dict): A dictionary mapping image names to image data.
        crop_coordinates (dict): A dictionary mapping image names to crop coordinates.
        coordinate (list): A list to store the coordinates of a point.

    Methods:
        load_images(): Loads the images from the image list.
        is_image_modified(image_name): Checks if an image has been modified.
        get_modified_images(): Returns the set of modified image names.
        set_coordinates(event, x, y, _, __): Sets the coordinates of a point.
        process_images(): Processes the images and extracts crop coordinates.
    """

    def __init__(self, image_list=[], image_names=[]):
        self.image_list = image_list
        self.image_names = image_names
        self.modified_images = set()
        self.image_dictionary, self.crop_coordinates = self.load_images()
        self.coordinate = None

    def load_images(self):
        """
        Loads the images from the image list.

        Returns:
            image_data (dict): A dictionary mapping image names to image data.
            coordinates (dict): A dictionary mapping image names to None.
        """
        image_data = []
        for image in self.image_list:
            image = Image.open(image)
            image_data.append(np.array(image))
        image_data = dict(zip(self.image_names, image_data))
        coordinates = dict(zip(self.image_names, [None] * len(self.image_names)))
        return image_data, coordinates

    def is_image_modified(self, image_name):
        """
        Checks if an image has been modified.

        Parameters:
            image_name (str): The name of the image.

        Returns:
            bool: True if the image has been modified, False otherwise.
        """
        return image_name in self.modified_images

    def get_modified_images(self):
        """
        Returns the set of modified image names.

        Returns:
            set: The set of modified image names.
        """
        return self.modified_images

    def set_coordinates(self, event, x, y, _, __):
        """
        Sets the coordinates of a point.

        Parameters:
            event: The event type.
            x (int): The x-coordinate of the point.
            y (int): The y-coordinate of the point.
            _:
            __:
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_image = self.original_image.copy()
            cv2.drawMarker(
                self.current_image,
                (x, y),
                (0, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            cv2.imshow(self.current_image_name, self.current_image)
            self.coordinate = [x, y]

    def process_images(self):
        """
        Processes the images and extracts crop coordinates.
        """
        total_images = len(self.image_dictionary)
        for image_name, image_data in tqdm(self.image_dictionary.items(), total=total_images, desc="Processing Images"):
            current_point = [None, None]
            if not image_name.lower().endswith((".png", ".jpg", ".jpeg")):
                print(f"[ERROR]: {image_name} is not a valid image file. Check the files supplied to the function.")
            elif self.is_image_modified(image_name):
                print(f"Skipping {image_name} as it has already been used.")
            else:
                self.current_image_name = image_name
                self.original_image = image_data  # Store the current image
                self.current_image = self.original_image.copy()
                cv2.namedWindow(self.current_image_name)
                cv2.moveWindow(self.current_image_name, 20, 20)
                cv2.setMouseCallback(self.current_image_name, self.set_coordinates)

                for i in range(2):
                    window_title = f"Click on the {'origin' if i == 0 else 'end'} point in image: {image_name}. Press any key to continue."
                    cv2.setWindowTitle(self.current_image_name, window_title)

                    while True:
                        cv2.imshow(self.current_image_name, self.current_image)
                        if cv2.waitKey(0):
                            current_point[i] = self.coordinate
                            break
                origin_x = current_point[0][0]
                origin_y = current_point[0][1]
                heigth = abs(current_point[0][1] - current_point[1][1])
                width = abs(current_point[0][0] - current_point[1][0])
                print(f"Origin: ({origin_x}, {origin_y})\nHeight: {heigth}\nWidth: {width}")
                self.modified_images.add(image_name)
                self.crop_coordinates[image_name] = (origin_x, origin_y, heigth, width)
                cv2.destroyAllWindows()
        print(f"Processed {len(self.modified_images)} images.")

class DataFiles:
    """
    This class organizes the files to be analyzed in separate dictionaries for each type of file
    Each dictionary has the animal name as key and the file path as value for each file

    Attributes:
        pos_file (dict): A dictionary containing the position files for each animal
        skeleton_file (dict): A dictionary containing the skeleton files for each animal
        jpg_file (dict): A dictionary containing the jpg files for each animal

    Methods:
        add_pos_file: Adds a position file to the pos_file dictionary
        add_skeleton_file: Adds a skeleton file to the skeleton_file dictionary
        add_jpg_file: Adds a jpg file to the jpg_file dictionary
    """

    def __init__(self):
        self.position_files = {}
        self.skeleton_files = {}
        self.experiment_images = {}
        self.roi_files = {}

    def add_pos_file(self, key, file):
        self.position_files[key] = file

    def add_skeleton_file(self, key, file):
        self.skeleton_files[key] = file

    def add_image_file(self, key, file):
        self.experiment_images[key] = file

    def add_roi_file(self, key, file):
        self.roi_files[key] = file

class Animal:
    """
    Class to store the data for each animal
    The data is separated in bodyparts, skeleton and experiment_jpg where each one is a dictionary
    containing keys that represent the data scheme from the deeplabcut analysis

    Attributes:
        name (str): The name of the animal
        experiment_jpg (str): The path to the experiment jpg file
        bodyparts (dict): A dictionary containing the bodyparts data for each animal
        skeleton (dict): A dictionary containing the skeleton data for each animal

    Methods:
        exp_dimensions: Returns the dimensions of the arena that the experiment was performed in
        exp_length: Returns the length of the experiment
    """

    def __init__(self):
        # Currently the initialization is hardcoding the bodyparts and skeleton names to mirror
        # the ones used in the test data.
        # See the TODO above to add an option to dinamically add bodypart name
        # TODO: Currently the initialization is hardcoding the bodyparts and skeleton names to mirror and throwing a message to the user:
        """
        "Bone {bone} not found in the skeleton file for the animal {self.name}"
        "Please check the name of the bone in the skeleton file"
        "The following bones are available:"
        "focinho_orelhae": [],
        "focinho_orelhad": [],
        "orelhad_orelhae": [],
        "orelhae_orelhad": [],
        "orelhad_centro": [],
        "orelhae_centro": [],
        "centro_rabo": [],
        """
        # This should automatically select the correct bones and assign them to the skeleton dictionary

        self.name = None
        self.animal_jpg = []
        self.position_file = []
        self.skeleton_file = []
        self.rois = [
            {
                "file": [],
                "x": [],
                "y": [],
                "width": [],
                "height": [],
            },
            {
                "file": [],
                "x": [],
                "y": [],
                "width": [],
                "height": [],
            },
            {
                "file": [],
                "x": [],
                "y": [],
                "width": [],
                "height": [],
            },
            {
                "file": [],
                "x": [],
                "y": [],
                "width": [],
                "height": [],
            },
        ]
        self.bodyparts = {
            "focinho": [],
            "orelhad": [],
            "orelhae": [],
            "centro": [],
            "rabo": [],
        }
        self.skeleton = {
            "focinho_orelhae": [],
            "focinho_orelhad": [],
            "orelhad_orelhae": [],
            "orelhae_orelhad": [],
            "orelhad_centro": [],
            "orelhae_centro": [],
            "centro_rabo": [],
        }

    def exp_dimensions(self):
        """
        __exp_dimensions__ Returns the dimensions of the arena that the experiment was performed in

        Args:
            data (DataFiles): DataFiles object containing the image file
            animal_name (str): The name of the animal that the image belongs to

        Returns:
            tuple: A tuple containing the dimensions of the image
        """
        try:
            image = self.animal_jpg
        except FileNotFoundError:
            print("Image file not found in animal object\nWas it loaded properly?\n")
            return None
        return image.shape

    def exp_length(self):
        """
        __exp_length__ Returns the length of the experiment

        Returns:
            int: An int containing the length of the experiment in frames
        """

        return len(self.bodyparts["focinho"]["x"])

    def add_roi(self, roi_file):
        rois = []
        [rois.append(key) for key in roi_file]
        for i, roi_path in enumerate(rois):
            roi_data = pd.read_csv(
                roi_path,
                sep=",",
            )
            self.rois[i]["file"] = roi_path
            self.rois[i]["x"] = roi_data["X"][0]
            self.rois[i]["y"] = roi_data["Y"][0]
            self.rois[i]["width"] = roi_data["Width"][0]
            self.rois[i]["height"] = roi_data["Height"][0]
        pass

    def add_bodypart(self, bodypart):
        """
        add_bodypart gets the data from the csv file and stores it in the bodyparts dictionary.
        Remember that, to extract the data from this csv file, as it has a header with 3 rows,
        the indexing method should be: dataframe.loc[:, ('bodypart', 'axis/likelihood')]

        Args:
            bodypart (str): A string containing the name of the bodypart to be added
            data (DataFiles): A DataFiles object containing the files to be analyzed
            animal_name (str): A string containing the name of the animal to be analyzed
        """
        extracted_data = pd.read_csv(
            self.position_file,
            sep=",",
            header=[1, 2],
            index_col=0,
            skip_blank_lines=False,
        )

        # The following line is necessary to convert the column names to lowercase
        # The data is stored in a MultiIndex dataframe, so the column names are tuples with the bodypart name and the axis/likelihood
        # The following line converts the tuples to lowercase strings

        extracted_data.columns = pd.MultiIndex.from_frame(extracted_data.columns.to_frame().map(str.lower))
        self.bodyparts[bodypart] = {
            "x": extracted_data[bodypart, "x"],
            "y": extracted_data[bodypart, "y"],
            "likelihood": extracted_data[bodypart, "likelihood"],
        }

    def add_skeleton(self, bone):
        """
        add_skeleton gets the data from the csv file and stores it in the skeleton dictionary.
        Remember that, to extract the data from this csv file, as it has a header with 3 rows,
        the indexing method should be: dataframe.loc[:, ('bodypart', 'axis/likelihood')] when
        indexin using loc

        Args:
            bone (str): A string containing the name of the bone to be added
            data (DataFiles): A DataFiles object containing the files to be analyzed
            animal_name (str): A string containing the name of the animal to be analyzed
        """
        extracted_data = pd.read_csv(
            self.skeleton_file,
            sep=",",
            header=[0, 1],
            index_col=0,
            skip_blank_lines=False,
        )

        # The following line is necessary to convert the column names to lowercase
        # The data is stored in a MultiIndex dataframe, so the column names are tuples with the bodypart name and the axis/likelihood
        # The following line converts the tuples to lowercase strings
        extracted_data.columns = pd.MultiIndex.from_frame(extracted_data.columns.to_frame().map(str.lower))
        try:
            self.skeleton[bone] = {
                "length": extracted_data[bone, "length"],
                "orientation": extracted_data[bone, "orientation"],
                "likelihood": extracted_data[bone, "likelihood"],
            }
        except KeyError:
            # print(f"\nBone {bone} not found in the skeleton file for the animal {self.name}")
            # print("Please check the name of the bone in the skeleton file\n")
            # print("The following bones are available:")
            # print("focinho_orelhae\nfocinho_orelhad\norelhad_orelhae\norelhae_orelhad\norelhae_rabo\norelhad_rabo\n")
            return

    def add_experiment_jpg(self, image_file):
        """
        add_experiment_jpg gets the data from the jpg file and stores it in the experiment_jpg attribute.

        Args:
            data (DataFiles): A DataFiles object containing the files to be analyzed
            animal_name (str): A string containing the name of the animal to be analyzed
        """
        try:
            raw_image = mpimg.imread(image_file)
            self.animal_jpg = raw_image
        except KeyError:
            print(f"\nJPG file for the animal {self.name} not found.\nPlease, check if the name of the file is correct.\n")
        return

    def add_position_file(self, position_file):
        """This function adds a reference to the position file to the animal object

        Args:
            position_file (csv): A csv file containing the position data for each bodypart of the animal
        """

        self.position_file = position_file

    def add_skeleton_file(self, skeleton_file):
        """This function adds a reference to the skeleton file to the animal object

        Args:
            skeleton_file (csv): A csv file containing the skeleton data for each bone created for the animal
        """

        self.skeleton_file = skeleton_file

    def get_jpg_dimensions(self):
        """
        get_jpg_dimensions returns the dimensions of the experiment's recording

        Returns:
            tuple: A tuple containing the dimensions of the jpg file
        """
        return self.animal_jpg.shape


def get_unique_names(file_list, regex):
    """
    get_unique_names generates a list of unique names from a list of files

    Args:
        file_list (list): A list of files containing the duplicated names to be reduced to unique names
        regex (object): A regex object to be used to extract the file names from the file list

    Returns:
        list: A list of unique names
    """
    unique_names = []

    for file in file_list:
        file_name = os.path.basename(file)
        if file_name.endswith(".jpg") or file_name.endswith(".png") or file_name.endswith(".jpeg"):
            try:
                unique_names.append(regex.search(file_name).group(0))
            except AttributeError:
                print(f"'{file_name}' not recognized")
                pass
        elif file_name.endswith(".csv"):
            try:
                unique_names.append(regex.search(file_name).group(0))
            except AttributeError:
                pass

    names = list(dict.fromkeys(unique_names))

    return names


def get_files(self, line_edit, data: DataFiles, animal_list: list, worker=None):
    """
    get_files organizes que files to be analyzed in separate dictionaries for each type of file
    where each dictionary has the animal name as key and the file path as value for each file

    Args:
        data (DataFiles): A DataFiles object to store the files in an organized way
        animal_list (list): An empty list of animal objects to be filled with the data from the files

    Returns:
        None: The function does not return anything, but it fills the data and animal_list objects
    """
    # debugpy.debug_this_thread()
    if worker is not None:
        worker.signals.request_files.emit("dlc_files")
        data_ready_event.wait()
        while not worker.stored_data:
            pass
        data_files = worker.stored_data[0]
        worker.stored_data = []
        data_ready_event.clear()
    else:
        data_files, _ = QFileDialog.getOpenFileNames(self, "Select the files to analyze", "", "DLC files (*.csv *.jpg)")

    get_name = re.compile(r"^.*?(?=DLC)|^.*?(?=(\.jpg|\.png|\.bmp|\.jpeg|\.svg))")
    # TODO #38 - Remove this regex and use the list created below to get the roi files4
    get_roi = re.compile(r"\b\w*roi[\w -~]*\.csv$")
    unique_animals = get_unique_names(data_files, get_name)
    roi_iter_obejct = it.filterfalse(lambda x: not (re.search("roi", x)), data_files)
    rois = []
    [rois.append(roi) for roi in roi_iter_obejct]

    for animal in unique_animals:
        for file in data_files:
            if animal in file:
                if "skeleton" in file and not data.skeleton_files.get(animal):
                    line_edit.append("Skeleton file found for " + animal)
                    data.add_skeleton_file(animal, file)
                    continue
                if "filtered" in file and not data.position_files.get(animal):
                    line_edit.append("Position file found for " + animal)
                    data.add_pos_file(animal, file)
                    continue
                if (file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")) and not data.experiment_images.get(animal):
                    line_edit.append("Image file found for " + animal)
                    data.add_image_file(animal, file)
                    continue
                if get_roi.search(file).group(0) in file and not data.roi_files.get(animal):
                    line_edit.append("ROI file found for " + animal)
                    data.add_roi_file(animal, file)
                    continue
    for exp_number, animal in enumerate(unique_animals):
        animal_list.append(Animal())
        animal_list[exp_number].name = animal
        animal_list[exp_number].add_experiment_jpg(data.experiment_images[animal])
        animal_list[exp_number].add_position_file(data.position_files[animal])
        animal_list[exp_number].add_skeleton_file(data.skeleton_files[animal])
        tmp = it.filterfalse(lambda roi: not (re.search(animal, roi)), rois)
        animal_list[exp_number].add_roi(tmp)

        for bodypart in animal_list[exp_number].bodyparts:
            animal_list[exp_number].add_bodypart(bodypart)
        for bone in animal_list[exp_number].skeleton:
            animal_list[exp_number].add_skeleton(bone)

    return data_files


def angle_between_lines(line1, line2, origin):
    """
    angle_between_lines Returns the angle between two lines given an origin

    Args:
        line1 (tuple): Two tuples containing the coordinates of the start and end points of the first line
        line2 (tuple): Two tuples containing the coordinates of the start and end points of the second line
        origin (tuple): A tuple containing the coordinates of the origin of the lines

    Returns:
        float: The angle between the two lines in degrees with the origin as the vertex
    """
    # Line segments are represented by tuples of two points
    # (x1, y1) -> start point of line segment 1
    # (x2, y2) -> end point of line segment 1
    # (x3, y3) -> start point of line segment 2
    # (x4, y4) -> end point of line segment 2
    x, y = origin

    x1, y1 = line1[0][0] - x, line1[0][1] - y
    x2, y2 = line1[1][0] - x, line1[1][1] - y
    x3, y3 = line2[0][0] - x, line2[0][1] - y
    x4, y4 = line2[1][0] - x, line2[1][1] - y

    # Calculate the dot product of the two vectors
    dot_product = x1 * x3 + y1 * y3 + x2 * x4 + y2 * y4

    # Calculate the magnitudes of the two vectors
    magnitude_a = math.sqrt(x1**2 + y1**2 + x2**2 + y2**2)
    magnitude_b = math.sqrt(x3**2 + y3**2 + x4**2 + y4**2)

    # Calculate the angle in radians
    angle = math.acos(dot_product / (magnitude_a * magnitude_b))

    # Convert to degrees and return
    return math.degrees(angle)


def line_trough_triangle_vertex(A, B, C):
    """
    line_trough_triangle_vertex Returns the points of a line passing through the center of the triangle's `A` vertex

    Args:
        A (tuple): A tuple containing the coordinates of the `A` vertex
        B (tuple): A tuple containing the coordinates of the `B` vertex
        C (tuple): A tuple containing the coordinates of the `C` vertex

    Returns:
        tuple: A tuple containing the coordinates of the start and end points of the line
    """

    # Compute the midpoint of the side opposite vertex A
    M = (B + C) / 2
    # Compute the vector from vertex A to the midpoint
    AM = M - A
    # Define the endpoints of the line passing through the center of vertex A
    P = A - 0.5 * AM
    Q = A + 0.1 * AM

    return P, Q


def sign(x):
    return -1 if x < 0 else 1


def detect_collision(line_segment_start, line_segment_end, circle_center, circle_radius):
    """Detects intersections between a line segment and a circle.

    Args:
        line_start (tuple): A tuple containing the (x, y) coordinates of the start point of the line segment.
        line_end (tuple): A tuple containing the (x, y) coordinates of the end point of the line segment.
        circle_center (tuple): A tuple containing the (x, y) coordinates of the center of the circle.
        circle_radius (float): The radius of the circle.

    Returns:
        list: A list of tuples representing the intersection points between the line segment and the circle.

    """

    line_start_x_relative_to_circle = line_segment_start[0] - circle_center[0]
    line_start_y_relative_to_circle = line_segment_start[1] - circle_center[1]
    line_end_x_relative_to_circle = line_segment_end[0] - circle_center[0]
    line_end_y_relative_to_circle = line_segment_end[1] - circle_center[1]
    line_segment_delta_x = line_end_x_relative_to_circle - line_start_x_relative_to_circle
    line_segment_delta_y = line_end_y_relative_to_circle - line_start_y_relative_to_circle

    line_segment_length = math.sqrt(line_segment_delta_x * line_segment_delta_x + line_segment_delta_y * line_segment_delta_y)
    discriminant_numerator = line_start_x_relative_to_circle * line_end_y_relative_to_circle - line_end_x_relative_to_circle * line_start_y_relative_to_circle
    discriminant = circle_radius * circle_radius * line_segment_length * line_segment_length - discriminant_numerator * discriminant_numerator
    if discriminant < 0:
        return []
    if discriminant == 0:
        intersection_point_1_x = (discriminant_numerator * line_segment_delta_y) / (line_segment_length * line_segment_length)
        intersection_point_1_y = (-discriminant_numerator * line_segment_delta_x) / (line_segment_length * line_segment_length)
        parameterization_a = (intersection_point_1_x - line_start_x_relative_to_circle) * line_segment_delta_x / line_segment_length + (
            intersection_point_1_y - line_start_y_relative_to_circle
        ) * line_segment_delta_y / line_segment_length
        return [(intersection_point_1_x + circle_center[0], intersection_point_1_y + circle_center[1])] if 0 < parameterization_a < line_segment_length else []

    intersection_point_1_x = (discriminant_numerator * line_segment_delta_y + sign(line_segment_delta_y) * line_segment_delta_x * math.sqrt(discriminant)) / (
        line_segment_length * line_segment_length
    )
    intersection_point_1_y = (-discriminant_numerator * line_segment_delta_x + abs(line_segment_delta_y) * math.sqrt(discriminant)) / (
        line_segment_length * line_segment_length
    )
    parameterization_a = (intersection_point_1_x - line_start_x_relative_to_circle) * line_segment_delta_x / line_segment_length + (
        intersection_point_1_y - line_start_y_relative_to_circle
    ) * line_segment_delta_y / line_segment_length
    intersection_points = [(intersection_point_1_x + circle_center[0], intersection_point_1_y + circle_center[1])] if 0 < parameterization_a < line_segment_length else []

    intersection_point_2_x = (discriminant_numerator * line_segment_delta_y - sign(line_segment_delta_y) * line_segment_delta_x * math.sqrt(discriminant)) / (
        line_segment_length * line_segment_length
    )
    intersection_point_2_y = (-discriminant_numerator * line_segment_delta_x - abs(line_segment_delta_y) * math.sqrt(discriminant)) / (
        line_segment_length * line_segment_length
    )
    parameterization_b = (intersection_point_2_x - line_start_x_relative_to_circle) * line_segment_delta_x / line_segment_length + (
        intersection_point_2_y - line_start_y_relative_to_circle
    ) * line_segment_delta_y / line_segment_length
    intersection_points += (
        [(intersection_point_2_x + circle_center[0], intersection_point_2_y + circle_center[1])] if 0 < parameterization_b < line_segment_length else []
    )
    return intersection_points


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


def folder_structure_check_function(self):
    self.interface.clear_unused_files_lineedit.clear()
    folder_path = os.path.dirname(self.interface.config_path_lineedit.text().replace('"', "").replace("'", ""))
    if folder_path == "":
        message = "Please, select a path to the config.yaml file before checking the folder structure."
        title = "Path to config file not selected"
        warning_message_function(title, message)

    required_folders = ["dlc-models", "evaluation-results", "labeled-data", "training-datasets", "videos"]
    required_files = ["config.yaml"]

    for folder in required_folders:
        if not os.path.isdir(os.path.join(folder_path, folder)):
            self.interface.clear_unused_files_lineedit.append(f"The folder '{folder}' is NOT present")
            return False
        self.interface.clear_unused_files_lineedit.append(f"The folder {folder} is OK")

    for file in required_files:
        if not os.path.isfile(os.path.join(folder_path, file)):
            self.interface.clear_unused_files_lineedit.append(f"The project's {file} is NOT present")
            return False
    # Check if dlc-models contains at least one iteration folder
    dlc_models_path = os.path.join(folder_path, "dlc-models")
    iteration_folders = [f for f in os.listdir(dlc_models_path) if os.path.isdir(os.path.join(dlc_models_path, f)) and f.startswith("iteration-")]
    if not iteration_folders:
        self.interface.clear_unused_files_lineedit.append("There are no iteration folders in dlc-models.")
        return False

    latest_iteration_folder = max(iteration_folders, key=lambda x: int(x.split("-")[1]))
    shuffle_set = os.listdir(os.path.join(dlc_models_path, latest_iteration_folder))
    if not shuffle_set:
        self.interface.clear_unused_files_lineedit.append("There are no shuffle sets in the latest iteration folder.")
        return False
    else:
        for root, dirs, files in os.walk(os.path.join(dlc_models_path, latest_iteration_folder, shuffle_set[0])):
            for dir in dirs:
                if dir.startswith("log"):
                    continue
                if "train" not in dirs or "test" not in dirs:
                    self.interface.clear_unused_files_lineedit.append("The train or test folder is missing.")
                    return False
                if dir.startswith("test") and not os.path.isfile(os.path.join(root, dir, "pose_cfg.yaml")):
                    self.interface.clear_unused_files_lineedit.append("The pose_cfg.yaml file is missing in test folder.")
                    return False
                if dir.startswith("train"):
                    if not os.path.isfile(os.path.join(root, dir, "pose_cfg.yaml")):
                        self.interface.clear_unused_files_lineedit.append("The pose_cfg.yaml file is missing in test folder.")
                        return False
                    elif not any("meta" in string for string in os.listdir(os.path.join(root, dir))):
                        self.interface.clear_unused_files_lineedit.append("The meta file is missing in train folder.")
                        return False
                    elif not any("data" in string for string in os.listdir(os.path.join(root, dir))):
                        self.interface.clear_unused_files_lineedit.append("The data file is missing in train folder.")
                        return False
                    elif not any("index" in string for string in os.listdir(os.path.join(root, dir))):
                        self.interface.clear_unused_files_lineedit.append("The index file is missing in train folder.")
                        return False

    # If all checks pass, the folder structure is correct
    self.interface.clear_unused_files_lineedit.append("The folder structure is correct.")
    return True


def dlc_video_analyze_function(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    # debugpy.debug_this_thread()
    request_files.emit("get_options")
    data_ready_event.wait()
    options = worker.stored_data[0]
    self.options = options
    worker.stored_data = []

    if self.interface.analyze_from_file_button.isEnabled():
        paths_file = self.interface.analyze_from_file_lineedit.text()
        videos_from_txt = [path[0] for path in pd.read_csv(paths_file, delimiter = "\t", header = None).values.tolist()]
        video_list = videos_from_txt
        config_path = self.interface.config_path_lineedit.text().replace('"', "").replace("'", "")
        if (config_path == "") or (video_list == ""):
            text_signal.emit(("Both the config file and the videos folder must be selected.", "clear_unused_files_lineedit"))
            return
        
        text_signal.emit(("clear_unused_files_lineedit", "clear_lineedit"))
        if DLC_ENABLE:
            text_signal.emit((f"Using DeepLabCut version {deeplabcut.__version__}", "clear_unused_files_lineedit"))

        file_extension = False
        valid_extensions = [".mp4", ".avi", ".mov"]
        invalid_files = [file for file in video_list if not any(file.endswith(ext) for ext in valid_extensions)]

        for file in video_list:
            if invalid_files:
                title = "Video extension error"
                message = "Videos must have the extension '.mp4', '.avi' or '.mov'.\n Please, check the videos folder and try again."
                warning_message.emit((title, message))
                return
            if (".mp4" in file or ".avi" in file or ".mov" in file) and (not file_extension):
                file_extension = file.split(".")[-1]
            elif file_extension and (file.split(".")[-1] != file_extension):
                title = "Video extension error"
                message = "All videos must have the same extension.\n Please, check the videos folder and try again."
                warning_message.emit((title, message))

        continue_analysis = self.resume_message_function(video_list)
        if not continue_analysis:
            text_signal.emit(("clear_lineedit", "clear_unused_files_lineedit"))
            text_signal.emit(("Analysis canceled.", "clear_unused_files_lineedit"))
            return
        text_signal.emit(("Analyzing videos...", "clear_unused_files_lineedit"))
        list_of_videos = [file for file in video_list]
        if DLC_ENABLE:
            deeplabcut.analyze_videos(config_path, list_of_videos, videotype=file_extension, shuffle=1, trainingsetindex=0, gputouse=0, allow_growth=True, save_as_csv=True)
        text_signal.emit(("Done analyzing videos.", "clear_unused_files_lineedit"))

        text_signal.emit(("Filtering data files and saving as CSV...", "clear_unused_files_lineedit"))
        if DLC_ENABLE:
            deeplabcut.filterpredictions(
                config_path,
                list_of_videos,
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

        if DLC_ENABLE:
            deeplabcut.plot_trajectories(
                config_path,
                list_of_videos,
                videotype=file_extension,
                showfigures=False,
                filtered=True,
            )
        # if DLC_ENABLE:
        #     os.rename(os.path.join(list_of_videos, "plot-poses"), os.path.join(list_of_videos, "accuracy_check_plots"))

        text_signal.emit(("Plots to visualize prediction accuracy were saved.", "clear_unused_files_lineedit"))
        text_signal.emit(("Done filtering data files", "clear_unused_files_lineedit"))
        self.options = {}
    else:
        text_signal.emit(("clear_unused_files_lineedit", "clear_lineedit"))
        if DLC_ENABLE:
            text_signal.emit((f"Using DeepLabCut version {deeplabcut.__version__}", "clear_unused_files_lineedit"))
        config_path = self.interface.config_path_lineedit.text().replace('"', "").replace("'", "")
        videos = self.interface.video_folder_lineedit.text().replace('"', "").replace("'", "")
        self.options["save_folder"] = videos
        if (config_path == "") or (videos == ""):
            text_signal.emit(("Both the config file and the videos folder must be selected.", "clear_unused_files_lineedit"))
            return
        video_list = [
            os.path.join(videos, file)
            for file in os.listdir(videos)
            if file.endswith(".mp4") or file.endswith(".avi") or file.endswith(".mov") or file.endswith(".mkv") or file.endswith(".flv") or file.endswith(".webm")
        ]
        file_extension = False
        valid_extensions = [".mp4", ".avi", ".mov"]
        invalid_files = [file for file in video_list if not any(file.endswith(ext) for ext in valid_extensions)]

        for file in video_list:
            if invalid_files:
                title = "Video extension error"
                message = "Videos must have the extension '.mp4', '.avi' or '.mov'.\n Please, check the videos folder and try again."
                warning_message.emit((title, message))
                return
            if (".mp4" in file or ".avi" in file or ".mov" in file) and (not file_extension):
                file_extension = file.split(".")[-1]
            elif file_extension and (file.split(".")[-1] != file_extension):
                title = "Video extension error"
                message = "All videos must have the same extension.\n Please, check the videos folder and try again."
                warning_message.emit((title, message))

        continue_analysis = self.resume_message_function(video_list)
        if not continue_analysis:
            text_signal.emit(("clear_lineedit", "clear_unused_files_lineedit"))
            text_signal.emit(("Analysis canceled.", "clear_unused_files_lineedit"))
            return
        text_signal.emit(("Analyzing videos...", "clear_unused_files_lineedit"))
        list_of_videos = [file for file in video_list]
        if DLC_ENABLE:
            deeplabcut.analyze_videos(config_path, list_of_videos, videotype=file_extension, shuffle=1, trainingsetindex=0, gputouse=0, allow_growth=True, save_as_csv=True)
        text_signal.emit(("Done analyzing videos.", "clear_unused_files_lineedit"))

        text_signal.emit(("Filtering data files and saving as CSV...", "clear_unused_files_lineedit"))
        if DLC_ENABLE:
            deeplabcut.filterpredictions(
                config_path,
                list_of_videos,
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

        if DLC_ENABLE:
            deeplabcut.plot_trajectories(
                config_path,
                list_of_videos,
                videotype=file_extension,
                showfigures=False,
                filtered=True,
            )
        if DLC_ENABLE:
            if not os.path.exists(os.path.join(videos, "accuracy_check_plots")):
                os.rename(os.path.join(videos, "plot-poses"), os.path.join(videos, "accuracy_check_plots"))
            else:
                text_signal.emit(("The accuracy_check_plots folder already exists. Skipping", "clear_unused_files_lineedit"))

        text_signal.emit(("Plots to visualize prediction accuracy were saved.", "clear_unused_files_lineedit"))
        text_signal.emit(("Done filtering data files", "clear_unused_files_lineedit"))
        self.options = {}


def get_frames_function(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    # debugpy.debug_this_thread()
    text_signal.emit(("clear_lineedit", "clear_unused_files_lineedit"))
    videos = self.interface.video_folder_lineedit.text().replace('"', "").replace("'", "")
    current_working_dir = os.path.dirname(__file__)
    task_duration = int(self.interface.task_duration_lineedit.text())
    fps = int(self.interface.frames_per_second_lineedit.text())
    where_to_extract = int((task_duration * fps) * 0.5)
    if self.interface.override_frame_extraction_checkbox.isChecked():
        where_to_extract = int(self.interface.override_frame_extraction_lineedit.text())
        if where_to_extract > task_duration * fps:
            title = "Frame extraction error"
            message = "The frame to be extracted is beyond the video duration.\n Please, check the frame to be extracted and try again."
            warning_message.emit((title, message))
            return

    if self.interface.analyze_from_file_button.isEnabled():
        paths_file = self.interface.analyze_from_file_lineedit.text()
        videos_from_txt = [path[0] for path in pd.read_csv(paths_file, delimiter = "\t", header = None).values.tolist()]
        video_list = videos_from_txt
    else:
        _, _, file_list = [entry for entry in os.walk(videos)][0]
        video_list = [os.path.join(videos, file) for file in os.listdir(videos) if file.endswith(".mp4") or file.endswith(".avi") or file.endswith(".mov")]
    
    file_extension = False
    valid_extensions = [".mp4", ".avi", ".mov"]
    invalid_files = [file for file in video_list if not any(file.endswith(ext) for ext in valid_extensions)]

    for file in video_list:
        if invalid_files:
            title = "Video extension error"
            message = "Videos must have the extension '.mp4', '.avi' or '.mov'.\n Please, check the videos folder and try again."
            warning_message.emit((title, message))
            return
        if (".mp4" in file or ".avi" in file or ".mov" in file) and (not file_extension):
            file_extension = file.split(".")[-1]
        elif file_extension and (file.split(".")[-1] != file_extension):
            title = "Video extension error"
            message = "All videos must have the same extension.\n Please, check the videos folder and try again."
            warning_message.emit((title, message))
    
    if self.interface.analyze_from_file_button.isEnabled():
        for filename in video_list:
            if filename.endswith(file_extension):
                output_path = os.path.splitext(filename)[0] + ".jpg"
                if not os.path.isfile(output_path):
                    text_signal.emit((f"Getting a frame of {filename}", "clear_unused_files_lineedit"))
                    subprocess.run(
                        f'ffmpeg -i "{filename}" -vf "select=eq(n\,{where_to_extract})" -fps_mode vfr -update 1 -frames:v 1 "{output_path}"',
                        shell=True,
                        cwd=os.path.join(current_working_dir, "ffmpeg", "bin"),
                    )
                else:
                    text_signal.emit((f"Frame of {filename} already exists.", "clear_unused_files_lineedit"))
    else:
        for filename in file_list:
            if filename.endswith(file_extension):
                video_path = os.path.join(videos, filename)
                output_path = os.path.splitext(video_path)[0] + ".jpg"
                if not os.path.isfile(output_path):
                    text_signal.emit((f"Getting a frame of {filename}", "clear_unused_files_lineedit"))
                    subprocess.run(
                        f'ffmpeg -i "{video_path}" -vf "select=eq(n\,{where_to_extract})" -fps_mode vfr -update 1 -frames:v 1 "{output_path}"',
                        shell=True,
                        cwd=os.path.join(current_working_dir, "ffmpeg", "bin"),
                    )
                else:
                    text_signal.emit((f"Frame of {filename} already exists.", "clear_unused_files_lineedit"))


def extract_skeleton_function(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    text_signal.emit(("clear_lineedit", "clear_unused_files_lineedit"))
    if DLC_ENABLE:
        text_signal.emit(("Using DeepLabCut version " + deeplabcut.__version__, "clear_unused_files_lineedit"))
        # self.interface.clear_unused_files_lineedit.append(f"Using DeepLabCut version {deeplabcut.__version__}")
    config_path = self.interface.config_path_lineedit.text().replace('"', "").replace("'", "")
    videos = self.interface.video_folder_lineedit.text().replace('"', "").replace("'", "")

    text_signal.emit(("Extracting skeleton...", "clear_unused_files_lineedit"))
    if self.interface.analyze_from_file_button.isEnabled():
        paths_file = self.interface.analyze_from_file_lineedit.text()
        videos_from_txt = [path[0] for path in pd.read_csv(paths_file, delimiter = "\t", header = None).values.tolist()]
        videos = videos_from_txt

    deeplabcut.filterpredictions(
        config_path,
        videos,
        videotype=".mp4",
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
    deeplabcut.analyzeskeleton(config_path, videos, shuffle=1, trainingsetindex=0, filtered=True, save_as_csv=True)
    text_signal.emit(("Done extracting skeleton.", "clear_unused_files_lineedit"))


def clear_unused_files_function(self):
    self.interface.clear_unused_files_lineedit.clear()
    videos = self.interface.video_folder_lineedit.text().replace('"', "").replace("'", "")

    if self.interface.analyze_from_file_button.isEnabled():
        paths_file = self.interface.analyze_from_file_lineedit.text()
        videos_from_txt = [path[0] for path in pd.read_csv(paths_file, delimiter = "\t", header = None).values.tolist()]
        videos = videos_from_txt

    unwanted_folder = os.path.join(videos, "unwanted_files")

    if not os.path.exists(unwanted_folder):
        os.makedirs(unwanted_folder)

    _, _, file_list = [entry for entry in os.walk(videos)][0]

    file_extension = ".mp4"
    for file in file_list:
        if any(ext in file for ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"]):
            file_extension = file.split(".")[-1]

    for file in file_list:
        file_path = os.path.join(videos, file)
        if file.endswith(file_extension) or file.endswith(".png") or file.endswith(".jpg") or file.endswith(".tiff") or "roi" in file:
            continue
        if file.endswith(".h5") or file.endswith(".pickle") or "filtered" not in file:
            shutil.move(file_path, os.path.join(unwanted_folder, file))
            self.interface.clear_unused_files_lineedit.append(f"Moved {file} to unwanted_files")

    _, _, file_list = [entry for entry in os.walk(videos)][0]

    has_filtered_csv = False
    has_skeleton_filtered_csv = False
    has_roi_file = False
    has_left_roi_file = False
    has_right_roi_file = False
    has_image_file = False
    missing_files = []
    task_type = self.interface.type_combobox.currentText().lower().strip().replace(" ", "_")

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
            if file.endswith("roiR.csv"):
                has_right_roi_file = True
                continue
            elif file.endswith("roiL.csv"):
                has_left_roi_file = True
                continue
        elif task_type == "social_recognition":
            if file.endswith("roi.csv"):
                has_roi_file = True
                continue
    if task_type == "social_recognition" and any(
        [
            not has_filtered_csv,
            not has_skeleton_filtered_csv,
            not has_roi_file,
            not has_image_file,
        ]
    ):
        self.interface.clear_unused_files_lineedit.append("There are missing files in the folder")
    elif task_type == "njr" and any(
        [
            not has_filtered_csv,
            not has_skeleton_filtered_csv,
            not has_left_roi_file,
            not has_right_roi_file,
            not has_image_file,
        ]
    ):
        self.interface.clear_unused_files_lineedit.append("There are missing files in the folder")
    else:
        self.interface.clear_unused_files_lineedit.append("All required files are present.")
        return
    if not has_filtered_csv:
        missing_files.append(" - filtered.csv")
    if not has_skeleton_filtered_csv:
        missing_files.append(" - skeleton_filtered.csv")
    if not has_image_file:
        missing_files.append(" - screenshot of the video")
    if task_type == "njr":
        if not has_left_roi_file:
            missing_files.append(" - roiR.csv")
        if not has_right_roi_file:
            missing_files.append(" - roiL.csv")
    if task_type == "social_recognition" and not has_roi_file:
        missing_files.append(" - roi.csv")

    title = "Missing files"
    message = "The following files are missing:\n\n" + "\n".join(
        missing_files + ["\nPlease, these files are essential for the analysis to work.\nCheck the analysis folder before continuing with the analysis."]
    )
    warning_message_function(title, message)


def get_folder_path_function(self, lineedit_name):
    if "config_path" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        config_file = str(Path(filedialog.askopenfilename(title="Select the config.yaml file", multiple=False)))
        self.interface.config_path_lineedit.setText(config_file)
    elif "videos_path" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the folder", mustexist=True)))
        self.interface.video_folder_lineedit.setText(folder)
    elif "config_path_data_process" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        config_file = str(Path(filedialog.askopenfilename(title="Select the config.yaml file", multiple=False)))
        self.interface.config_path_data_process_lineedit.setText(config_file)
    elif "videos_path_data_process" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the folder", mustexist=True)))
        self.interface.video_folder_data_process_lineedit.setText(folder)
    elif "crop_path" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the folder", mustexist=True)))
        self.interface.videos_to_crop_folder_lineedit.setText(folder)
    elif "crop_path_video_editing" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the folder", mustexist=True)))
        self.interface.videos_to_crop_folder_video_editing_lineedit.setText(folder)
    elif "source_folder" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the source folder", mustexist=True)))
        self.interface.source_folder_path_video_editing_lineedit.setText(folder)
    elif "destination_folder" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the destination folder", mustexist=True)))
        self.interface.destination_folder_path_video_editing_lineedit.setText(folder)
    elif "file_to_analyze" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        file = str(Path(filedialog.askopenfilename(title="Select the file to analyze", multiple=False)))
        self.interface.analyze_from_file_lineedit.setText(file)
    elif "create_roi_automatically" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the folder", mustexist=True)))
        self.interface.folder_to_get_create_roi_lineedit.setText(folder)
    elif "get_bout_analysis_folder" == lineedit_name.lower():
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call("wm", "attributes", ".", "-topmost", True)
        folder = str(Path(filedialog.askdirectory(title="Select the folder", mustexist=True)))
        self.interface.path_to_bout_analysis_folder_lineedit.setText(folder)


def check_roi_files(roi):
    extracted_data = pd.read_csv(roi, sep=",")
    must_have = ["x", "y", "width", "height"]
    header = extracted_data.columns.to_frame().map(str.lower).to_numpy()
    return all(elem in header for elem in must_have)


def create_frequency_grid(x_values, y_values, bin_size, analysis_range, *extra_data):
    """
    Creates a frequency grid based on the given x and y values.

    Args:
        x_values (list): List of x-coordinate values.
        y_values (list): List of y-coordinate values.
        bin_size (float): Size of each bin in the grid.
        analysis_range (tuple): Range of indices to consider for analysis.
        *extra_data: Additional data (e.g., speed, mean_speed).

    Returns:
        numpy.ndarray: The frequency grid.

    """
    if extra_data:
        speed = extra_data[0]
        mean_speed = extra_data[1]

    # Calculate a gridmap with an exploration heatmap
    xy_values = [(int(x_values[i]), int(y_values[i])) for i in range(analysis_range[0], analysis_range[1])]

    # Find the minimum and maximum values of x and y
    min_x = int(min(x_values))
    max_x = int(max(x_values))
    min_y = int(min(y_values))
    max_y = int(max(y_values))

    # Calculate the number of bins in each dimension
    num_bins_x = int((max_x - min_x) / bin_size) + 1
    num_bins_y = int((max_y - min_y) / bin_size) + 1

    # Create a grid to store the frequencies
    grid = np.zeros((num_bins_y, num_bins_x), dtype=int)
    if extra_data:
        # Assign the values to their corresponding bins in the grid
        for ii, xy in enumerate(xy_values):
            xi, yi = xy
            bin_x = (xi - min_x) // bin_size
            bin_y = (yi - min_y) // bin_size
            if speed[ii] > mean_speed:
                grid[bin_y, bin_x] += 1
    else:
        # Assign the values to their corresponding bins in the grid
        for xy in xy_values:
            xi, yi = xy
            bin_x = (xi - min_x) // bin_size
            bin_y = (yi - min_y) // bin_size
            grid[bin_y, bin_x] += 1  # Increment the frequency of the corresponding bin

    return grid


def options_to_configuration(configuration):
    # Define mapping for key conversions
    options_to_configuration = {
        "algo_type": "Algorithm Type",
        "arena_height": "Arena height",
        "arena_width": "Arena width",
        "crop_video": "Crop video",
        "experiment_type": "Experiment Type",
        "frames_per_second": "Video framerate",
        "max_fig_res": "Plot resolution",
        "plot_options": "Plot option",
        "save_folder": "Saved data folder",
        "task_duration": "Task Duration",
        "threshold": "Experimental Animal",
        "trim_amount": "Amount to trim",
    }

    # Perform key conversions
    converted_data = {}
    for old_key, new_key in options_to_configuration.items():
        if old_key in configuration:
            converted_data[new_key] = configuration[old_key]

    # Add or modify specific keys
    if converted_data["Experimental Animal"] == 0.0267:
        converted_data["Experimental Animal"] = "mouse"
    elif converted_data["Experimental Animal"] == 0.0667:
        converted_data["Experimental Animal"] = "rat"

    return converted_data


def configuration_to_options(configuration):
    configuration_to_options = {
        "Algorithm Type": "algo_type",
        "Arena height": "arena_height",
        "Arena width": "arena_width",
        "Crop video": "crop_video",
        "Experiment Type": "experiment_type",
        "Video framerate": "frames_per_second",
        "Plot resolution": "max_fig_res",
        "Plot option": "plot_options",
        "Saved data folder": "save_folder",
        "Task Duration": "task_duration",
        "Experimental Animal": "threshold",
        "Amount to trim": "trim_amount",
    }

    # Perform key conversions
    converted_data = {}
    for old_key, new_key in configuration_to_options.items():
        if old_key in configuration:
            converted_data[new_key] = configuration[old_key]

    # Add or modify specific keys
    if converted_data["threshold"] == "mouse":
        converted_data["threshold"] = 0.0267
    elif converted_data["threshold"] == "rat":
        converted_data["threshold"] = 0.0667

    return converted_data


def test_configuration_file(config_path):
    try:
        with open(config_path, "r") as file:
            configuration = json.load(file)
            return configuration
    except:
        return False


def file_selection_function(self):
    file_dialog = QFileDialog(self.interface)
    file_dialog.setNameFilter("JSON files (*.json)")
    file_dialog.setWindowTitle("Select a configuration file")
    if file_dialog.exec():
        return file_dialog.selectedFiles()[0]


def run_analysis(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    line_edit = self.interface.resume_lineedit
    data = DataFiles()
    selected_folder_to_save = 0
    experiments = []
    selected_files = get_files(self, line_edit, data, experiments, worker)

    request_files.emit("save_folder")
    data_ready_event.wait()
    selected_folder_to_save = worker.stored_data[0]
    worker.stored_data = []
    data_ready_event.clear()

    try:
        assert selected_folder_to_save != ""
        assert len(experiments) > 0
    except AssertionError:
        if len(selected_files) == 0:
            line_edit.append(" ERROR: No files selected")
            return
        else:
            line_edit.append(" ERROR: No destination folder selected")
            return

    request_files.emit("get_options")
    data_ready_event.wait()
    options = worker.stored_data[0]
    self.options = options
    self.options["save_folder"] = selected_folder_to_save
    worker.stored_data = []
    data_ready_event.clear()

    results_data_frame = pd.DataFrame()
    for i, experiment in enumerate(experiments):
        analysis_results, data_frame = video_analyse(self, options, experiment)
        results_data_frame = results_data_frame.join(data_frame) if not results_data_frame.empty else data_frame
        progress.emit(round(((i + 1) / len(experiments)) * 100))
        if options["plot_options"] == "plotting_enabled":
            plot_analysis_social_behavior(experiment, analysis_results, options)
    return results_data_frame, options


def handle_results(results):
    # debugpy.debug_this_thread()
    if results[0] is None:
        print("Done!")
    else:
        try:
            results_data_frame, options = results[0]
            results_data_frame.T.to_excel(options["save_folder"] + "/analysis_results.xlsx")
        except:
            os.system("cls")
            print("Error saving results")


def handle_error(error_info):
    exctype, value, traceback_str = error_info
    print(f"Error: {value}")


def on_worker_finished(self):
    options = self.options
    if options == {}:
        return
    config_file_path = options["save_folder"] + "/analysis_configuration.json"
    config_json = options_to_configuration(options)
    with open(config_file_path, "w") as config_file:
        json.dump(config_json, config_file, indent=4, sort_keys=True)
    text = "Analysis completed successfully."
    title = "Analysis completed"
    warning_message_function(title, text)


def clear_interface(self):
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
    self.interface.crop_video_time_lineedit.setText("0")
    self.interface.crop_video_checkbox.setChecked(False)
    self.interface.fig_max_size.setCurrentIndex(1)
    self.interface.plot_data_checkbox.setChecked(True)


def load_configuration_file(self, configuration={}):
    if configuration:
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
            clear_interface(self)
            return

        if configuration["Algorithm Type"].lower().strip().replace(" ", "_") == "deeplabcut":
            self.interface.algo_type_combobox.setCurrentIndex(0)
        elif configuration["Algorithm Type"].lower().strip().replace(" ", "_") == "bonsai":
            self.interface.algo_type_combobox.setCurrentIndex(1)
        else:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid algorithm type.")
            clear_interface(self)
            return

        if configuration["Arena width"] > 0 and configuration["Arena width"] < 650:
            self.interface.arena_width_lineedit.setText(str(int(configuration["Arena width"])))
        else:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid arena width.")
            clear_interface(self)
            return

        if configuration["Arena height"] > 0 and configuration["Arena height"] < 650:
            self.interface.arena_height_lineedit.setText(str(int(configuration["Arena height"])))
        else:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid arena height.")
            clear_interface(self)
            return

        if configuration["Video framerate"] > 0 and configuration["Video framerate"] < 240:
            self.interface.frames_per_second_lineedit.setText(str(int(configuration["Video framerate"])))
        else:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid number of frames per second.")
            clear_interface(self)
            return

        if configuration["Experimental Animal"].lower().strip().replace(" ", "_") == "mouse":
            self.interface.animal_combobox.setCurrentIndex(0)
        elif configuration["Experimental Animal"].lower().strip().replace(" ", "_") == "rat":
            self.interface.animal_combobox.setCurrentIndex(1)
        else:
            warning_message_function(
                "Configuration file",
                "The file selected is not a valid configuration file.\n Please, select a valid experimental animal.\n We support mice and rats at the time.",
            )
            clear_interface(self)
            return

        if configuration["Task Duration"] > 0 and configuration["Task Duration"] < 86400:
            self.interface.task_duration_lineedit.setText(str(int(configuration["Task Duration"])))
        else:
            warning_message_function(
                "Configuration file",
                "The file selected is not a valid configuration file.\n Please, select a valid task duration. We support up to 24 hours of task duration.",
            )
            clear_interface(self)
            return

        if configuration["Amount to trim"] < 0:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a positive trim amount.")
            clear_interface(self)
            return
        elif configuration["Amount to trim"] > configuration["Task Duration"]:
            warning_message_function(
                "Configuration file", "The file selected is not a valid configuration file.\n Please, select a trim amount smaller than the task duration."
            )
            clear_interface(self)
            return
        else:
            self.interface.crop_video_time_lineedit.setText(str(int(configuration["Amount to trim"])))

        resolutions = [["640", "480"], ["1280", "720"], ["1920", "1080"], ["2560", "1440"]]
        if configuration["Plot resolution"] in resolutions:
            self.interface.fig_max_size.setCurrentIndex(resolutions.index(configuration["Plot resolution"]))
        else:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid figure resolution.")
            clear_interface(self)
            return

        if configuration["Plot option"] == "plotting_enabled":
            self.interface.plot_data_checkbox.setChecked(True)
        elif configuration["Plot option"] == "plotting_disabled":
            self.interface.plot_data_checkbox.setChecked(False)
        else:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid plot option.")
            clear_interface(self)
            return

        if configuration["Crop video"] == True:
            self.interface.crop_video_checkbox.setChecked(True)
        elif configuration["Crop video"] == False:
            self.interface.crop_video_checkbox.setChecked(False)
        else:
            warning_message_function("Configuration file", "The file selected is not a valid configuration file.\n Please, select a valid crop video option.")
            clear_interface(self)
            return
    else:
        warning_message_function("The configuratuin file is empty, setting the default values.")
        clear_interface(self)
        return


def get_experiments(self, line_edit, recent_analysis_folder_line_edit, algo_type="deeplabcut"):
    if algo_type == "deeplabcut":
        data = DataFiles()
        inexistent_file = 0
        selected_folder_to_save = 0
        error = 0
        experiments = []
        selected_files = get_files(self, line_edit, data, experiments)
        selected_folder_to_save = QFileDialog.getExistingDirectory(self, "Select the folder to save the plots")
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
    recent_analysis_folder_line_edit.setText(selected_folder_to_save)
    return experiments, selected_folder_to_save, error, inexistent_file


def video_analyse(self, options, animal=None):
    if options["algo_type"] == "deeplabcut":
        collision_data = []
        dimensions = animal.exp_dimensions()
        focinho_x = animal.bodyparts["focinho"]["x"]
        focinho_y = animal.bodyparts["focinho"]["y"]
        orelha_esq_x = animal.bodyparts["orelhae"]["x"]
        orelha_esq_y = animal.bodyparts["orelhae"]["y"]
        orelha_dir_x = animal.bodyparts["orelhad"]["x"]
        orelha_dir_y = animal.bodyparts["orelhad"]["y"]
        centro_x = animal.bodyparts["centro"]["x"]
        centro_y = animal.bodyparts["centro"]["y"]
        roi_X = []
        roi_Y = []
        roi_D = []
        roi_NAME = []
        roi_regex = re.compile(r"\/([^\/]+)\.")
        number_of_filled_rois = sum(1 for roi in animal.rois if roi["x"])
        for i in range(number_of_filled_rois):
            # Finds the name of the roi in the file name
            roi_NAME.append(roi_regex.search(animal.rois[i]["file"]).group(1))
            roi_X.append(animal.rois[i]["x"])
            roi_Y.append(animal.rois[i]["y"])
            roi_D.append((animal.rois[i]["width"] + animal.rois[i]["height"]) / 2)
        # ---------------------------------------------------------------

        # General data
        arena_width = options["arena_width"]
        arena_height = options["arena_height"]
        frames_per_second = options["frames_per_second"]
        max_analysis_time = options["task_duration"]
        threshold = options["threshold"]
        # Maximum video height set by user
        # (height is stored in the first element of the list and is converted to int beacuse it comes as a string)
        max_video_height = int(options["max_fig_res"][0])
        max_video_width = int(options["max_fig_res"][1])
        trim_amount = int(options["trim_amount"] * frames_per_second)
        video_height, video_width, _ = dimensions
        factor_width = arena_width / video_width
        factor_height = arena_height / video_height
        number_of_frames = animal.exp_length()
        bin_size = 10
        # ----------------------------------------------------------------------------------------------------------
        if options["crop_video"]:
            runtime = range(trim_amount, int((max_analysis_time * frames_per_second) + trim_amount))
            if number_of_frames < max(runtime):
                runtime = range(trim_amount, int(number_of_frames))
                # TODO: #42 Add a warning message when the user sets a trim amount that is too high.
                print("\n")
                print(f"---------------------- WARNING FOR ANIMAL {animal.name} ----------------------")
                print(
                    f"The trim amount set is too high for the animal {animal.name}.\nThe analysis for this video will be done from {int(trim_amount/frames_per_second)} to {max_analysis_time} seconds."
                )
                print(
                    f"At the end of the analysis you should check if the analysis is coherent with the animal's behavior.\nIf it's not, redo the analysis with a lower trim amount for this animal: {animal.name}"
                )
                print("-------------------------------------------------------------------------------\n")
        else:
            runtime = range(int(max_analysis_time * frames_per_second))
        for i in runtime:
            # Calculate the area of the mice's head
            Side1 = np.sqrt(((orelha_esq_x[i] - focinho_x[i]) ** 2) + ((orelha_esq_y[i] - focinho_y[i]) ** 2))
            Side2 = np.sqrt(((orelha_dir_x[i] - orelha_esq_x[i]) ** 2) + ((orelha_dir_y[i] - orelha_esq_y[i]) ** 2))
            Side3 = np.sqrt(((focinho_x[i] - orelha_dir_x[i]) ** 2) + ((focinho_y[i] - orelha_dir_y[i]) ** 2))
            S = (Side1 + Side2 + Side3) / 2
            mice_head_area = np.sqrt(S * (S - Side1) * (S - Side2) * (S - Side3))
            # ------------------------------------------------------------------------------------------------------

            # Calculate the exploration threshold in front of the mice's nose
            A = np.array([focinho_x[i], focinho_y[i]])
            B = np.array([orelha_esq_x[i], orelha_esq_y[i]])
            C = np.array([orelha_dir_x[i], orelha_dir_y[i]])
            P, Q = line_trough_triangle_vertex(A, B, C)
            # ------------------------------------------------------------------------------------------------------

            # Calculate the collisions between the ROI and the mice's nose
            for ii in range(number_of_filled_rois):
                collision = detect_collision([Q[0], Q[1]], [P[0], P[1]], [roi_X[ii], roi_Y[ii]], roi_D[ii] / 2)
                if collision:
                    collision_data.append([1, collision, mice_head_area, roi_NAME[ii]])
                else:
                    collision_data.append([0, None, mice_head_area, None])

        # ----------------------------------------------------------------------------------------------------------
        corrected_runtime_last_frame = runtime[-1] + 1
        corrected_first_frame = runtime[1] - 1
        ANALYSIS_RANGE = [corrected_first_frame, corrected_runtime_last_frame]
        # ----------------------------------------------------------------------------------------------------------
        collisions = pd.DataFrame(collision_data)
        xy_data = collisions[1].dropna()

        # The following line substitutes these lines:
        #   t = xy_data.to_list()
        #   t = [item for sublist in t for item in sublist]
        #   x, y = zip(*t)
        # Meaning that it flattens the list and then separates the x and y coordinates
        try:
            x_collision_data, y_collision_data = zip(*[item for sublist in xy_data.to_list() for item in sublist])
        except ValueError:
            x_collision_data, y_collision_data = np.zeros(len(runtime)), np.zeros(len(runtime))  # If there is no collision, the x and y collision data will be 0
            print("\n")
            print(f"---------------------- WARNING FOR ANIMAL {animal.name} ----------------------")
            print(f"Something went wrong with the animal's {animal.name} exploration data.\nThere are no exploration data in the video for this animal.")
            print(f"Please check the video for this animal: {animal.name}")
            print("-------------------------------------------------------------------------------\n")

        # ----------------------------------------------------------------------------------------------------------

        # Calculate the total exploration time
        exploration_mask = collisions[0] > 0
        exploration_mask = exploration_mask.astype(int)
        exploration_time = np.sum(exploration_mask) * (1 / frames_per_second)

        # Calculate the total exploration time in each ROI
        filtered_mask_right = collisions[collisions.iloc[:, -1].fillna("").str.contains("roiR")]
        filtered_mask_left = collisions[collisions.iloc[:, -1].fillna("").str.contains("roiL")]
        count_right = len(filtered_mask_right)
        count_left = len(filtered_mask_left)
        exploration_time_right = count_right * (1 / frames_per_second)
        exploration_time_left = count_left * (1 / frames_per_second)

        x_axe = centro_x[corrected_first_frame:corrected_runtime_last_frame]  # Raw x position data
        y_axe = centro_y[corrected_first_frame:corrected_runtime_last_frame]  # Raw y position data

        x_axe_cm = centro_x[corrected_first_frame:corrected_runtime_last_frame] * factor_width  # Puts the x position on scale
        y_axe_cm = centro_y[corrected_first_frame:corrected_runtime_last_frame] * factor_height  # Puts the y position on scale
        # The runtime[1]-1 is accessing the second element in the runtime list and subtracting 1. This is
        # done to adjust for the fact that Python indexing starts at 0.
        # So we are going from the start of the experiment, set by runtime with or without the trim amount
        # and going to the end of the experiment, set by corrected_runtime_last_frame

        # Calculates the step difference of position in x axis
        d_x_axe_cm = np.append(0, np.diff(x_axe_cm))
        # Calculates the step difference of position in y axis
        d_y_axe_cm = np.append(0, np.diff(y_axe_cm))

        displacement_raw = np.sqrt(np.square(d_x_axe_cm) + np.square(d_y_axe_cm))
        displacement = displacement_raw
        displacement[displacement < threshold] = 0
        displacement[displacement > 55] = 0

        # Sums all the animal's movements and calculates the accumulated distance traveled
        accumulate_distance = np.cumsum(displacement)

        # Gets the animal's total distance traveled
        total_distance = max(accumulate_distance)
        time_vector = np.linspace(0, len(runtime) / frames_per_second, len(runtime))  # Creates a time vector

        # Ignores the division by zero at runtime
        # (division by zero is not an error in this case as the are moments when the animal is not moving)
        np.seterr(divide="ignore", invalid="ignore")
        # Calculates the first derivative and finds the animal's velocity per time
        velocity = np.divide(displacement, np.transpose(np.append(0, np.diff(time_vector))))
        mean_velocity = np.nanmean(velocity)

        # Calculates the animal's acceleration
        acceleration = np.divide(np.append(0, np.diff(velocity)), np.append(0, np.diff(time_vector)))
        # Calculates the number of movements made by the animal
        movements = np.sum(displacement > 0)
        # Calculates the total time of movements made by the animal
        time_moving = np.sum(displacement > 0) * (1 / frames_per_second)
        # Calculates the total time of the animal without movimentations
        time_resting = np.sum(displacement == 0) * (1 / frames_per_second)

        kde_space_coordinates = np.array([np.array(x_axe), np.array(y_axe)])
        kde_instance = stats.gaussian_kde(kde_space_coordinates)
        point_density_function = kde_instance.evaluate(kde_space_coordinates)
        color_limits = np.array(
            [(x - np.min(point_density_function)) / (np.max(point_density_function) - np.min(point_density_function)) for x in point_density_function]
        )

        movement_points = np.array([x_axe, y_axe]).T.reshape(-1, 1, 2)
        # Creates a 2D array containing the line segments coordinates
        movement_segments = np.concatenate([movement_points[:-1], movement_points[1:]], axis=1)
        filter_size = 4
        moving_average_filter = np.ones((filter_size,)) / filter_size
        smooth_segs = np.apply_along_axis(lambda m: np.convolve(m, moving_average_filter, mode="same"), axis=0, arr=movement_segments)

        # Creates a LineCollection object with custom color map
        movement_line_collection = LineCollection(smooth_segs, cmap="plasma", linewidth=1.5)
        # Set the line color to the normalized values of "color_limits"
        movement_line_collection.set_array(velocity)

        position_grid = create_frequency_grid(focinho_x, focinho_y, bin_size, ANALYSIS_RANGE)
        velocity_grid = create_frequency_grid(centro_x, centro_y, bin_size, ANALYSIS_RANGE, velocity, mean_velocity)

        analysis_results = {
            "y_pos_data": y_axe,
            "x_pos_data": x_axe,
            "x_collision_data": x_collision_data,
            "y_collision_data": y_collision_data,
            "exploration_time": exploration_time,
            "exploration_time_right": exploration_time_right,
            "exploration_time_left": exploration_time_left,
            "position_grid": position_grid,
            "velocity_grid": velocity_grid,
            "velocity": velocity,
            "video_width": video_width,
            "video_height": video_height,
            "max_video_height": max_video_height,
            "max_video_width": max_video_width,
            "dimensions": animal.exp_dimensions(),
            "focinho_x": animal.bodyparts["focinho"]["x"],
            "focinho_y": animal.bodyparts["focinho"]["y"],
            "orelha_esq_x": animal.bodyparts["orelhae"]["x"],
            "orelha_esq_y": animal.bodyparts["orelhae"]["y"],
            "orelha_dir_x": animal.bodyparts["orelhad"]["x"],
            "orelha_dir_y": animal.bodyparts["orelhad"]["y"],
            "roi_X": animal.rois[0]["x"],
            "roi_Y": animal.rois[0]["y"],
            "roi_D": (animal.rois[0]["width"] + animal.rois[0]["height"]) / 2,
            "collision_data": collision_data,
            "color_limits": color_limits,
            "accumulate_distance": accumulate_distance,
            "total_distance": total_distance,
            "movements": movements,
            "acceleration": acceleration,
            "displacement": displacement,
            "time_moving": time_moving,
            "time_resting": time_resting,
            "mean_velocity": mean_velocity,
            "analysis_range": ANALYSIS_RANGE,
        }
        if options["experiment_type"] == "njr":
            dict_to_excel = {
                "exploration_time (s)": exploration_time,
                "exploration_time_right (s)": exploration_time_right,
                "exploration_time_left (s)": exploration_time_left,
                "time moving (S)": time_moving,
                "time resting (S)": time_resting,
                "total distance (cm)": total_distance,
                "mean velocity (cm/s)": mean_velocity,
            }
        elif options["experiment_type"] == "social_recognition":
            dict_to_excel = {
                "exploration_time (s)": exploration_time,
                "time moving (s)": time_moving,
                "time resting (s)": time_resting,
                "total distance (cm)": total_distance,
                "mean velocity (cm/s)": mean_velocity,
            }
        data_frame = pd.DataFrame(data=dict_to_excel, index=[animal.name])
        data_frame = data_frame.T
        self.interface.resume_lineedit.append(f"Analysis for animal {animal.name} completed.")
        return analysis_results, data_frame


def plot_analysis_social_behavior(experiment, analysis_results, options):
    animal_image = experiment.animal_jpg
    animal_name = experiment.name
    save_folder = options["save_folder"]
    x_pos = analysis_results["x_pos_data"]
    y_pos = analysis_results["y_pos_data"]
    image_height = analysis_results["video_height"]
    image_width = analysis_results["video_width"]
    max_height = analysis_results["max_video_height"]
    max_width = analysis_results["max_video_width"]
    x_collisions = analysis_results["x_collision_data"]
    y_collisions = analysis_results["y_collision_data"]
    position_grid = analysis_results["position_grid"]
    accumulate_distance = analysis_results["accumulate_distance"]
    frames_per_second = options["frames_per_second"]
    ANALYSIS_RANGE = analysis_results["analysis_range"]
    analysis_time_frames = ANALYSIS_RANGE[1] - ANALYSIS_RANGE[0]
    time_vector_secs = np.arange(0, analysis_time_frames / frames_per_second, 1 / frames_per_second)

    # Calculate the ratio to be used for image resizing without losing the aspect ratio
    ratio = min(max_height / image_width, max_width / image_height)
    new_resolution_in_inches = (int(image_width * ratio / 100), int(image_height * ratio / 100))

    with plt.ioff():
        fig_1, axe_1 = plt.subplots()
        fig_1.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
        fig_1.set_size_inches(new_resolution_in_inches)
        axe_1.set_title(
            "Overall heatmap of the mice's nose position", loc="center", fontdict={"fontsize": new_resolution_in_inches[1] * 2, "fontweight": "normal", "color": "black"}
        )
        axe_1.set_xlabel("X (pixels)", fontdict={"fontsize": new_resolution_in_inches[1] * 1.2, "fontweight": "normal", "color": "black"})
        axe_1.set_ylabel("Y (pixels)", fontdict={"fontsize": new_resolution_in_inches[1] * 1.2, "fontweight": "normal", "color": "black"})
        axe_1.set_xticks([])
        axe_1.set_yticks([])

        fig_2, axe_2 = plt.subplots()
        fig_2.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
        fig_2.set_size_inches(new_resolution_in_inches)
        axe_2.set_title("Exploration map by ROI", loc="center", fontdict={"fontsize": new_resolution_in_inches[1] * 2, "fontweight": "normal", "color": "black"})
        axe_2.set_xticks([])
        axe_2.set_yticks([])

        fig_4, axe_4 = plt.subplots()
        fig_4.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
        fig_4.set_size_inches(new_resolution_in_inches)
        axe_4.set_title("Distance accumulated over time", loc="center", fontdict={"fontsize": new_resolution_in_inches[1] * 2, "fontweight": "normal", "color": "black"})
        axe_4.set_xlabel("Time (s)", fontdict={"fontsize": new_resolution_in_inches[1] * 1.2, "fontweight": "normal", "color": "black"})
        axe_4.set_ylabel("Distance (cm)", fontdict={"fontsize": new_resolution_in_inches[1] * 1.2, "fontweight": "normal", "color": "black"})
        axe_4.grid(True)

        fig_5, axe_5 = plt.subplots()
        fig_5.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, hspace=0, wspace=0)
        fig_5.set_size_inches(new_resolution_in_inches)
        axe_5.set_title("Animal movement in the arena", loc="center", fontdict={"fontsize": new_resolution_in_inches[1] * 2, "fontweight": "normal", "color": "black"})
        axe_5.axis("off")

        # Plot the heatmap of the mice's nose position
        axe_1.imshow(position_grid, cmap="inferno", interpolation="bessel")
        fig_1.savefig(save_folder + "/" + animal_name + " Overall heatmap of the mice's nose position.png")

        # Plot the Overall exploration by ROI
        kde_axis = sns.kdeplot(x=x_collisions, y=y_collisions, ax=axe_2, cmap="inferno", fill=True, alpha=0.5)
        axe_2.imshow(animal_image, interpolation="bessel")
        fig_2.savefig(save_folder + "/" + animal_name + " Overall exploration by ROI.png")

        # Plot the distance accumulated over time
        axe_4.plot(time_vector_secs, accumulate_distance)
        fig_4.savefig(save_folder + "/" + animal_name + " Distance accumulated over time.png")

        # Plot the animal movement in the arena
        axe_5.plot(x_pos, y_pos, color="orangered", linewidth=1.5)
        axe_5.imshow(animal_image, interpolation="bessel", alpha=0.9)
        fig_5.savefig(save_folder + "/" + animal_name + " Animal movement in the arena.png")

        plt.close("all")


def option_message_function(self, text, info_text):
    """
    Displays a custom dialog box with the given text and info_text.
    Returns "yes" if the user accepts the dialog, "no" otherwise.
    """
    dialog = CustomDialog(text, info_text, self.interface)
    result = dialog.exec()

    if result == QDialog.Accepted:
        return "yes"
    else:
        return "no"


def convert_csv_to_h5(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    """
    Converts CSV files to H5 format using DeepLabCut.

    This function takes the values from the `config_path_lineedit` and `scorer_lineedit` fields of the `interface` object.
    It also checks the value of the `confirm_folders_checkbox` to determine whether to prompt the user for confirmation.

    If the `config` field is empty, an error message is displayed in the `log_lineedit` field and the function returns.
    If the `scorer` field is empty, an error message is displayed in the `log_lineedit` field and the function returns.

    Parameters:
        self (object): The instance of the class containing the function.

    Returns:
        None
    """

    config = self.interface.config_path_data_process_lineedit.text()
    scorer = self.interface.scorer_data_process_lineedit.text()
    confirm_folders = self.interface.confirm_folders_checkbox.isChecked()

    if config == "":
        text_signal.emit(("[ERROR]: Please select a config file.", "log_data_process_lineedit"))
        return
    elif scorer == "":
        text_signal.emit(("[ERROR]: Please select a scorer.", "log_data_process_lineedit"))
        return
    deeplabcut.convertcsv2h5(config, userfeedback=confirm_folders, scorer=scorer)


def analyze_folder_with_frames(sworker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    """
    Analyzes a folder containing video frames using DeepLabCut.

    This function prompts the user to select a config file and a video folder. It then displays a message with the list
    of files in the video folder and asks the user to confirm if they want to proceed with the analysis. If the user
    confirms, it calls the `analyze_time_lapse_frames` function from DeepLabCut to perform pose inference on the frames.

    Args:
        self: The instance of the class containing the function.

    Returns:
        None
    """
    config = self.interface.config_path_data_process_lineedit.text()
    video_folder = self.interface.video_folder_data_process_lineedit.text()
    frametype = self.interface.frames_extensions_combobox.currentText().strip().lower()

    if config == "":
        text_signal.emit(("[ERROR]: Please select a config file.", "clear_unused_files_lineedit"))
        return
    elif video_folder == "":
        text_signal.emit(("[ERROR]: Please select a folder.", "clear_unused_files_lineedit"))
        return

    deeplabcut.analyze_time_lapse_frames(config, video_folder, frametype, save_as_csv=True)
    deeplabcut.filterpredictions(
        config,
        video_folder,
        videotype=frametype,
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
    deeplabcut.plot_trajectories(config, video_folder, videotype=frametype, showfigures=False)


def get_crop_coordinates(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    """
    Get dimensions to crop videos based on the provided folder path.

    This function takes the folder path from the `videos_to_crop_folder_lineedit` text field of the interface,
    retrieves the image names from the folder, and creates a list of image paths. It then initializes a `video_editing_tool`
    object with the image list and names, and processes the images to obtain the crop coordinates. The crop coordinates
    are stored in the `crop_coordinates` attribute of the object.

    Finally, the function iterates over the `crop_coordinates` dictionary and appends the crop coordinates for each image
    to the `log_lineedit_page_2` text field of the interface.

    note: The `crop_coordinates` attribute is stored in the `self` object for later use in ffmpeg cropping.

    """
    folder_path = self.interface.videos_to_crop_folder_video_editing_lineedit.text()
    image_names = [file for file in os.listdir(folder_path) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    video_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith((".mp4", ".avi", ".mov")) and not "_cropped" in file]
    if os.path.exists(os.path.join(folder_path, "crop_coordinates.csv")):
        with open(os.path.join(folder_path, "crop_coordinates.csv"), mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader)
            temp_coordinates = {rows[0]: [int(rows[1]), int(rows[2]), int(rows[3]), int(rows[4])] for rows in reader}

        if len(set(video_list)) == len(set(temp_coordinates)):
            if set(image_names) == set(temp_coordinates.keys()):
                text_signal.emit(("[INFO]: Crop coordinates found. Loading from csv.", "log_video_editing_lineedit"))
                # self.interface.log_video_editing_lineedit.append("[INFO]: Crop coordinates found. Loading from csv.")
                self.crop_coordinates = temp_coordinates
                return
    else:
        text_signal.emit(("[ERROR]: No crop coordinates found. Defaulting to manual cropping.", "log_video_editing_lineedit"))
        # self.interface.log_video_editing_lineedit.append("[ERROR]: The saved csv does not contain crop coordinates for all images. Defaulting to manual cropping.")
        image_list = [os.path.join(folder_path, image) for image in image_names]
        crop_tool = video_editing_tool(image_list, image_names)
        crop_tool.process_images()
        coordinates = crop_tool.crop_coordinates
        self.crop_coordinates = coordinates
    return


def crop_videos(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    """
    Crop videos based on the provided crop coordinates.

    This function takes the folder path containing videos to be cropped and the crop coordinates
    stored in the `self.coordinates` dictionary. It crops each video using the corresponding
    crop coordinates and saves the cropped videos in the same folder.

    """
    # debugpy.debug_this_thread()
    cropped = set()
    folder_path = self.interface.videos_to_crop_folder_video_editing_lineedit.text()
    image_names = [file for file in os.listdir(folder_path) if file.lower().endswith((".png", ".jpg", ".jpeg"))]
    video_list = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith((".mp4", ".avi", ".mov")) and not "_cropped" in file]
    current_working_dir = os.path.dirname(__file__)
    if len(set(video_list)) != len(set(image_names)):
        text_signal.emit(
            ("[ERROR]: There is a mismatch between the video and image files. Please ensure that all videos have corresponding images.", "log_video_editing_lineedit")
        )
        # self.interface.log_video_editing_lineedit.append(
        #     "[ERROR]: There is a mismatch between the video and image files. Please ensure that all videos have corresponding images."
        # )
        return
    dictionary_without_extension_in_name = {key.split(".")[0]: value for key, value in self.crop_coordinates.items()}

    for image_name, crop_coordinates in self.crop_coordinates.items():
        text_signal.emit((f"Crop Coordinates for {image_name}: {crop_coordinates}", "log_video_editing_lineedit"))
        # self.interface.log_video_editing_lineedit.append(f"Crop Coordinates for {image_name}: {crop_coordinates}")

    for video_path in video_list:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if video_name in cropped:
            text_signal.emit((f"[INFO]: {video_name} has already been cropped.", "log_video_editing_lineedit"))
            # self.interface.log_video_editing_lineedit.append(f"[INFO]: {video_name} has already been cropped.")
            continue

        if video_name in dictionary_without_extension_in_name:
            origin_x, origin_y, width, height = dictionary_without_extension_in_name[video_name]
            out_file = os.path.join(folder_path, f"{video_name}_cropped.mp4")
            subprocess.run(
                f'ffmpeg -i "{video_path}" -filter:v "crop={width}:{height}:{origin_x}:{origin_y}" "{out_file}"',
                shell=True,
                cwd=os.path.join(current_working_dir, "ffmpeg", "bin"),
            )
            cropped.add(video_name)
            text_signal.emit((f"[INFO]: Cropped video {video_name} successfully.", "log_video_editing_lineedit"))
            # self.interface.log_video_editing_lineedit.append(f"[INFO]: Cropped video {video_name} successfully.")
        else:
            text_signal.emit((f"[WARNING]: No crop coordinates found for video {video_name}.", "log_video_editing_lineedit"))
            # self.interface.log_video_editing_lineedit.append(f"[WARNING]: No crop coordinates found for video {video_name}.")


def copy_folder_robocopy(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    source = self.interface.source_folder_path_video_editing_lineedit.text()
    destination = self.interface.destination_folder_path_video_editing_lineedit.text()
    # If the destination path contains spaces, change the directory name to use underscores
    if " " in destination:
        warning_message.emit(("Destination folder", "The destination folder path contains spaces. Please remove the spaces and try again."))
        return

    exclude_files = self.interface.exclude_files_checkbox.isChecked()

    if source == "":
        text_signal.emit(("[ERROR]: Please select a source folder.", "log_video_editing_lineedit"))
        # self.interface.source_folder_path_video_editing_lineedit.append("[ERROR]: Please select a source folder.")
        return
    elif destination == "":
        text_signal.emit(("[ERROR]: Please select a destination folder.", "log_video_editing_lineedit"))
        # self.interface.source_folder_path_video_editing_lineedit.append("[ERROR]: Please select a destination folder.")
        return

    if exclude_files:
        extensions_to_exclude = self.interface.file_exclusion_video_editing_lineedit.text().lower().strip().split(",")
        extensions_str = " ".join([f'"*{ext.strip()}"' for ext in extensions_to_exclude])
        command = f'robocopy "{source}" "{destination}" /e /zb /copyall /xf {extensions_str}'
    else:
        command = f'robocopy "{source}" "{destination}" /e /zb /copyall'

    # Prepare the command to run as administrator
    admin_command = f"powershell -Command \"Start-Process cmd -ArgumentList '/c {command}' -Verb RunAs\""

    try:
        subprocess.run(admin_command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        text_signal.emit((f"[ERROR]: {str(e)}", "log_video_editing_lineedit"))
        self.interface.log_video_editing_lineedit.append(f"[ERROR]: {str(e)}")


def save_crop_coordinates(self):
    """
    Saves the crop coordinates to a CSV file.

    If no crop coordinates are found, an error message is displayed.
    """
    if self.crop_coordinates is None:
        self.interface.log_video_editing_lineedit.append("[ERROR]: No crop coordinates found.")
        return
    folder_path = self.interface.videos_to_crop_folder_video_editing_lineedit.text()
    crop_coordinates = self.crop_coordinates
    csv_file_path = os.path.join(folder_path, "crop_coordinates.csv")

    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "x_origin", "y_origin", "width", "height"])
        for name, coordinate in crop_coordinates.items():
            writer.writerow([name] + list(coordinate))
    self.interface.log_video_editing_lineedit.append(f"[INFO]: Crop coordinates saved to {csv_file_path}.")


def create_rois_automatically(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    """
    Automatically creates ROIs (Region of Interest) for images in a specified folder using Deeplabcut.
    Args:
        text_signal (function, optional): Signal to emit text messages. Defaults to None.
        progress (function, optional): Signal to emit progress updates. Defaults to None.
        warning_message (function, optional): Signal to emit warning messages. Defaults to None.
        resume_message (function, optional): Signal to emit resume messages. Defaults to None.
    Returns:
        None
    """

    roi_network_config = os.path.join(os.path.dirname(__file__), "roi_network", "config.yaml")
    folder_with_analyzed_files = self.interface.folder_to_get_create_roi_lineedit.text()
    if folder_with_analyzed_files == "":
        text_signal.emit(("[ERROR]: Please select a folder.", "log_video_editing_lineedit"))
        return

    images_in_folder = [os.path.join(folder_with_analyzed_files, file) for file in os.listdir(folder_with_analyzed_files) if file.endswith(".jpg")]
    videos_in_folder = [os.path.join(folder_with_analyzed_files, file) for file in os.listdir(folder_with_analyzed_files) if file.endswith(".mp4")]

    images_in_folder_names = [file for file in os.listdir(folder_with_analyzed_files) if file.endswith(".jpg")]

    image_not_in_tmp_folder = []
    without_roi = []

    if len(images_in_folder) != len(videos_in_folder):
        text_signal.emit((f"Number of PNGs and videos in the folder do not match: {len(images_in_folder)} PNGs and {len(videos_in_folder)} videos.", "log_create_roi_lineedit"))
        image_names = [file for file in os.listdir(folder_with_analyzed_files) if file.endswith(".jpg")]
        video_names = [file for file in os.listdir(folder_with_analyzed_files) if file.endswith(".mp4")]
        if not all([name in video_names for name in image_names]):
            text_signal.emit(("The following files are missing:", "log_video_editing_lineedit"))
            text_signal.emit(([name for name in image_names if name not in video_names], "log_video_editing_lineedit"))

    config_path = roi_network_config
    tmp_folder = os.path.join(folder_with_analyzed_files, "tmp")

    try:
        os.mkdir(tmp_folder)
    except FileExistsError:
        text_signal.emit(("Temporary folder already exists.", "log_video_editing_lineedit"))

    # Check if all images are in the tmp folder
    images_in_tmp_folder = [file for file in os.listdir(tmp_folder) if file.endswith(".jpg")]
    roi_files_in_tmp_folder = [file for file in os.listdir(tmp_folder) if file.endswith("_roi.csv")]
    if not ((len(images_in_tmp_folder) == len(roi_files_in_tmp_folder))) or not (len(images_in_tmp_folder) == len(images_in_folder)):
        for image in images_in_folder_names:
            name = image.split(".")[0]
            if f"{name}.jpg" not in images_in_tmp_folder:
                image_not_in_tmp_folder.append(image)
        if image_not_in_tmp_folder:
            text_signal.emit(("Copying missing/unanalyzed images to tmp folder.", "log_video_editing_lineedit"))
            for image in image_not_in_tmp_folder:
                shutil.copy(os.path.join(folder_with_analyzed_files, image), os.path.join(tmp_folder, image))

    images_in_tmp_folder = [file for file in os.listdir(tmp_folder) if file.endswith(".jpg")]
    for image in images_in_tmp_folder:
        if not os.path.exists(os.path.join(tmp_folder, f"{image.split('.')[0]}_roi.csv")):
            without_roi.append(image)

    # Use Deeplabcut to extract circle coordinates
    dlc_files = [os.path.join(tmp_folder, file) for file in os.listdir(tmp_folder) if "DLC_resnet" in file]
    extensions = [ext.split(".")[-1] for ext in dlc_files]
    analyze = False
    if not any([ext in "h5" for ext in extensions]):
        analyze = True
    elif image_not_in_tmp_folder:
        analyze = True

    if analyze:
        for file in dlc_files:
            os.remove(file)
        deeplabcut.analyze_time_lapse_frames(config_path, tmp_folder, ".jpg", save_as_csv=True)
    # Search the folder for the h5 file generated by Deeplabcut
    h5_file = [file for file in os.listdir(tmp_folder) if file.endswith(".h5")][0]

    coordinates_list = pd.read_hdf(os.path.join(tmp_folder, h5_file))
    columns_to_drop = [col for col in coordinates_list.columns if "likelihood" in col]
    coordinates_list_clean = coordinates_list.drop(columns=columns_to_drop)

    # Loop through each image and create a generate a csv with the x, y, width and height of the circle
    for coordinates, image_name in zip(coordinates_list_clean.values, coordinates_list_clean.index.to_list()):
        if image_name in without_roi:
            x_coords_list = [coordinates[0], coordinates[2], coordinates[4], coordinates[6]]
            y_coords_list = [coordinates[1], coordinates[3], coordinates[5], coordinates[7]]

            x_mean = np.mean(x_coords_list)
            y_mean = np.mean(y_coords_list)

            diagonal = np.sqrt(np.square(x_coords_list[2] - x_coords_list[0]) + np.square(y_coords_list[2] - y_coords_list[0]))
            radius = int(diagonal/2)
            center = (int(x_mean), int(y_mean))

            file_name = image_name.split(".")[0]

            csv_data = {
                'X': x_mean,
                'Y': y_mean,
                'BX': int(x_mean),
                'BY': int(y_mean),
                'Width': radius * 2,
                'Height': radius * 2
            }

            roi_dataframe = pd.DataFrame(csv_data, index=[0])
            file = os.path.join(tmp_folder, f"{file_name}_roi.csv")
            if os.path.exists(file):
                text_signal.emit((f"Roi: {file} already exists.", "log_video_editing_lineedit"))
                continue
            roi_dataframe.to_csv(file, index=False)

    # Move the roi files to the folder with the analyzed files
    roi_files_in_tmp_folder = [file for file in os.listdir(tmp_folder) if file.endswith("_roi.csv")]
    for roi_file in roi_files_in_tmp_folder:
        if os.path.exists(os.path.join(folder_with_analyzed_files, roi_file)):
            text_signal.emit((f"Roi file {roi_file} already exists.", "log_video_editing_lineedit"))
            continue
        shutil.copy(os.path.join(tmp_folder, roi_file), folder_with_analyzed_files)

    # Check the ROI for each image
    images_in_tmp_folder = [file for file in os.listdir(tmp_folder) if file.endswith(".jpg")]
    for image_file in images_in_tmp_folder:
        roi_file = os.path.join(tmp_folder, f"{image_file.split('.')[0]}_roi.csv")
        if not os.path.exists(roi_file):
            text_signal.emit((f"Roi for image {image_file} was not found.", "log_video_editing_lineedit"))
            continue
        roi_data = pd.read_csv(roi_file)
        center = (int(roi_data['X'][0]), int(roi_data['Y'][0]))
        radius = int(roi_data['Width'][0] / 2)
        try:
            image = cv2.imread(os.path.join(tmp_folder, image_file))
        except Exception as e:
            text_signal.emit((f"Error reading image {image_file}.", "log_video_editing_lineedit"))
            text_signal.emit(("The error occurred: when using cv2.imread."))
            text_signal.emit(("The image might be corrupted."))        
            print(e)
            continue
        thickness = 1

        fig, ax = plt.subplots()
        ax.imshow(image)
        circle = plt.Circle(center, radius, color='red', fill=False, linewidth=thickness)
        ax.add_patch(circle)  # Add the circle to the axes
        ax.axis('off')
        ax.set_title(f"ROI for {image_file}")
        fig.set_tight_layout('tight')

        roi_check_folder = os.path.join(folder_with_analyzed_files, "auto_roi_check")
        if not os.path.exists(roi_check_folder):
            os.mkdir(roi_check_folder)
        image = os.path.join(roi_check_folder, f"{image_file.split('.')[0]}_roi_check.jpg")
        if os.path.exists(image):
            text_signal.emit((f"Image to check the ROI for {image_file} already exists.", "log_video_editing_lineedit"))
            continue
        else:
            text_signal.emit((f"Saving image {image_file}_roi_check.jpg", "log_video_editing_lineedit"))
            fig.savefig(image)
        plt.close(fig)


def bout_analysis(worker, self, text_signal=None, progress=None, warning_message=None, resume_message=None, request_files=None):
    analysis_folder = self.interface.path_to_bout_analysis_folder_lineedit.text()
    request_files.emit("get_options")
    data_ready_event.wait()
    options = worker.stored_data[0]
    self.options = options
    options["save_folder"] = self.interface.path_to_bout_analysis_folder_lineedit.text()
    worker.stored_data = []
    data_ready_event.clear()

    analysis_folder = self.interface.path_to_bout_analysis_folder_lineedit.text()
    figures_folder = os.path.join(analysis_folder, "bout_analysis")
    if not os.path.exists(figures_folder):
        os.mkdir(figures_folder)
    
    data = DataFiles()
    animals = []
    all_results = {}
    
    get_files(self, [], data, animals, worker)

    for animal in animals:
        collision_data = []
        # dimensions = animal.exp_dimensions()
        focinho_x = animal.bodyparts["focinho"]["x"]
        focinho_y = animal.bodyparts["focinho"]["y"]
        orelha_esq_x = animal.bodyparts["orelhae"]["x"]
        orelha_esq_y = animal.bodyparts["orelhae"]["y"]
        orelha_dir_x = animal.bodyparts["orelhad"]["x"]
        orelha_dir_y = animal.bodyparts["orelhad"]["y"]
        centro_x = animal.bodyparts["centro"]["x"]
        centro_y = animal.bodyparts["centro"]["y"]
        roi_X = []
        roi_Y = []
        roi_D = []
        roi_NAME = []
        number_of_filled_rois = sum(1 for roi in animal.rois if roi["x"])
        for i in range(number_of_filled_rois):
            roi_name = Path(animal.rois[i]["file"]).stem.split("_")[0]
            roi_NAME.append(roi_name)
            roi_X.append(animal.rois[i]["x"])
            roi_Y.append(animal.rois[i]["y"])
            roi_D.append((animal.rois[i]["width"] + animal.rois[i]["height"]) / 2)
        # ---------------------------------------------------------------

        # General data
        # arena_width = options["arena_width"]
        # arena_height = options["arena_height"]
        frames_per_second = options["frames_per_second"]
        max_analysis_time = options["task_duration"]
        # threshold = options["threshold"]
        # max_video_height = int(options["max_fig_res"][0])
        # max_video_width = int(options["max_fig_res"][1])
        # plot_options = True
        # trim_amount = int(options["trim_amount"] * frames_per_second)
        # video_height, video_width, _ = dimensions
        # factor_width = arena_width / video_width
        # factor_height = arena_height / video_height
        # number_of_frames = animal.exp_length()

        # ----------------------------------------------------------------------------------------------------------

        runtime = range(int(max_analysis_time * frames_per_second))

        # Try to clean the animal name from the date and time timestamp
        clean_animal_name = None
        try:
            # Replace the date and time timestamp with the an empty string
            # Replace all remaining underscores, spcaces and dashes with a single underscore
            new_key = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", "", animal.name)
            new_key = re.sub(r"[_\s-]+", "_", new_key)
            clean_animal_name = new_key
        except:
            text_signal.emit((f"Could not find the date and time timestamp for the animal {animal.name}", "log_data_process_lineedit"))
            text_signal.emit(("Using the complete animal name instead.", "log_data_process_lineedit"))
            clean_animal_name = animal.name

        for i in runtime:
            # Calculate the area of the mice's head
            Side1 = np.sqrt(((orelha_esq_x[i] - focinho_x[i]) ** 2) + ((orelha_esq_y[i] - focinho_y[i]) ** 2))
            Side2 = np.sqrt(((orelha_dir_x[i] - orelha_esq_x[i]) ** 2) + ((orelha_dir_y[i] - orelha_esq_y[i]) ** 2))
            Side3 = np.sqrt(((focinho_x[i] - orelha_dir_x[i]) ** 2) + ((focinho_y[i] - orelha_dir_y[i]) ** 2))
            S = (Side1 + Side2 + Side3) / 2
            mice_head_area = np.sqrt(S * (S - Side1) * (S - Side2) * (S - Side3))
            # ------------------------------------------------------------------------------------------------------

            # Calculate the exploration threshold in front of the mice's nose
            A = np.array([focinho_x[i], focinho_y[i]])
            B = np.array([orelha_esq_x[i], orelha_esq_y[i]])
            C = np.array([orelha_dir_x[i], orelha_dir_y[i]])
            P, Q = line_trough_triangle_vertex(A, B, C)
            # ------------------------------------------------------------------------------------------------------

            # Calculate the collisions between the ROI and the mice's nose
            for ii in range(number_of_filled_rois):
                collision = detect_collision([Q[0], Q[1]], [P[0], P[1]], [roi_X[ii], roi_Y[ii]], roi_D[ii] / 2)
                if collision:
                    collision_data.append([1, collision, mice_head_area, roi_NAME[ii]])
                else:
                    collision_data.append([0, None, mice_head_area, None])

        # ----------------------------------------------------------------------------------------------------------
        ## TODO: #43 If there is no collision, the collision_data will be empty and the code will break. Throw an error and print a message to the user explaining what is a collision.
        collisions = pd.DataFrame(collision_data)
        xy_data = collisions[1].dropna()

        # The following line substitutes these lines:
        #   t = xy_data.to_list()
        #   t = [item for sublist in t for item in sublist]
        #   x, y = zip(*t)
        # Meaning that it flattens the list and then separates the x and y coordinates
        try:
            x_collision_data, y_collision_data = zip(*[item for sublist in xy_data.to_list() for item in sublist])
        except ValueError:
            x_collision_data, y_collision_data = np.zeros(len(runtime)), np.zeros(len(runtime))  # If there is no collision, the x and y collision data will be 0
            print("\n")
            print(f"---------------------- WARNING FOR ANIMAL {animal.name} ----------------------")
            print(f"Something went wrong with the animal's {animal.name} exploration data.\nThere are no exploration data in the video for this animal.")
            print(f"Please check the video for this animal: {animal.name}")
            print("-------------------------------------------------------------------------------\n")

        # ----------------------------------------------------------------------------------------------------------

        # Calculate the total exploration time
        exploration_mask = collisions[0] > 0
        exploration_mask = exploration_mask.astype(int)

        # get sections
        exploration_bouts = find_sections(collisions, frames_per_second)
        exploration_bouts = exploration_bouts[exploration_bouts["duration"] > (1 / frames_per_second) * 2].reset_index(drop=True)
        # Save excel file with the exploration bouts
        exploration_bouts_sec = exploration_bouts.copy(deep=True)
        exploration_bouts_sec["start"] = exploration_bouts_sec["start"] / frames_per_second
        exploration_bouts_sec["end"] = exploration_bouts_sec["end"] / frames_per_second
        exploration_bouts_sec = exploration_bouts_sec.astype(float).round(2)
        exploration_bouts_sec = exploration_bouts_sec.rename(columns={"start": "start (s)", "end": "end (s)", "duration": "duration (s)"})
        exploration_bouts_sec.to_excel(os.path.join(figures_folder, f"{clean_animal_name}_exploration_bouts.xlsx"))
        durations = np.array(exploration_bouts["duration"])

        # Normalize durations and map to colors
        cmap = plt.get_cmap("afmhot")
        half_cmap = mcolors.LinearSegmentedColormap.from_list(f"half_{cmap.name}", cmap(np.linspace(0.1, 0.5, 256)))
        norm = mcolors.Normalize(vmin=durations.min(), vmax=durations.max(), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=half_cmap)
        colors = [mapper.to_rgba(duration) for duration in durations]
        x_positions = np.linspace(0, len(runtime) / frames_per_second, len(runtime))

        plt.close("all")
        fig, ax = plt.subplots(2, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1, 0.025]})
        [axi[1].set_axis_off() for axi in ax]
        plt.subplots_adjust(wspace=0.1, hspace=1)

        # Calculate the center positions for each line
        mid_points = [(start + end) / 2 for start, end in zip(exploration_bouts["start"], exploration_bouts["end"])]

        # Create horizontal event lines at y positions [0, 1, 2, ..., len(durations)-1]
        y_offset = 0
        positions = np.zeros(len(durations)) + y_offset
        transposed_positions = (np.array(positions)[:, np.newaxis]) / frames_per_second
        transposed_mid_points = np.array(mid_points)[:, np.newaxis] / frames_per_second
        x_points_sec = len(x_positions) / frames_per_second

        # Plot event lines
        ax[0][0].eventplot(transposed_mid_points, orientation="horizontal", colors="k", lineoffsets=transposed_positions, linelengths=durations)
        ax[0][0].set_xlim(0, x_points_sec)
        ax[0][0].set_ylim(y_offset - max(durations), y_offset + max(durations))
        ax[0][0].set_xlabel("Time (s)")
        ax[0][0].legend(["Exploration bouts"], loc="upper right", prop={"size": 6})
        ax[0][0].set_title(f"Duration mapped by size")

        ax[1][0].eventplot(transposed_mid_points, orientation="horizontal", colors=colors, lineoffsets=transposed_positions, linelengths=max(durations), linewidths=durations)
        ax[1][0].set_xlim(0, x_points_sec)
        ax[1][0].set_ylim(y_offset - max(durations), y_offset + max(durations))
        ax[1][0].set_xlabel("Time (s)")
        ax[1][0].legend(["Exploration bouts"], loc="upper right", prop={"size": 6})
        ax[1][0].set_title(f"Duration mapped by color and linewidth")

        for axi in [ax[1]]:
            cbar = fig.colorbar(mapper, ax=axi[1], location="right", ticks=np.around(np.linspace(0, max(durations), 6), 1))
            cbar.set_label("Duration (s)")
        [ax[0].set_yticks([]) for ax in ax]
        fig.text(0.10, 0.5, f"Exploration bouts for animal {clean_animal_name}", va="center", rotation="vertical", fontsize=14)
        fig.savefig(os.path.join(figures_folder, f"{clean_animal_name}_exploration_bouts.png"))

        plt.close("all")
        fig, ax = plt.subplots(figsize=(13, 4))
        unique, counts = np.unique(np.round(durations, decimals=1), return_counts=True)
        histogram_values = dict(zip(unique, counts))
        positions = range(len(unique))
        xticklabels = [str(val) for val in unique]
        ax.set_axisbelow(True)
        ax.grid(color="gray", linestyle="dashed", alpha=0.2, zorder=0)
        ax.bar(positions, counts, color="r", alpha=0.5, zorder=3)
        ax.set_title(f"Exploration bout duration histogram for animal {clean_animal_name}")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Frequency")
        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels, rotation=60)
        fig.tight_layout()
        
        fig.savefig(os.path.join(figures_folder, f"{clean_animal_name}_exploration_bouts_histogram.png"))

        plt.close("all")
        fig, ax = plt.subplots(figsize=(18, 4))
        ax.plot(x_positions, np.zeros(len(x_positions)), color="k", alpha=0.8, linewidth=10)
        for index, row in exploration_bouts.iterrows():
            ax.plot(
                [row["start"] / 30, row["end"] / 30],
                np.zeros(len([row["start"], row["end"]])),
                color="lightsalmon",
                alpha=0.5,
                linewidth=10,
                label="Exploration bouts" if index == 0 else "",
            )
        ax.set_title(f"Exploration bouts - Time series - Animal {clean_animal_name}")
        ax.set_xlabel("Time (s)")
        ax.set_yticks([0])
        ax.set_yticklabels([animal.name])
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_folder, f"{clean_animal_name}_exploration_bouts_time_series.png"))

        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 10))
        analysis_runtime = int(max_analysis_time * frames_per_second)
        plt.plot(centro_x.iloc[0:analysis_runtime], centro_y.iloc[0:analysis_runtime], label="x coordinates")
        for _, row in exploration_bouts.iterrows():
            start_idx = int(row["start"])
            end_idx = int(row["end"])
            was_inside = False
            for x, y in zip(centro_x[start_idx:end_idx], centro_y[start_idx:end_idx]):
                if is_inside_circle(x, y, roi_X[0], roi_Y[0], roi_D[0]):
                    was_inside = True
                    break
            if was_inside:
                continue
            plt.plot(centro_x[start_idx:end_idx], centro_y[start_idx:end_idx], color="red", linewidth=2)
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title(f"2D Plot of Coordinates with Highlighted Segments for Animal {clean_animal_name}")
        plt.tight_layout()

        fig.savefig(os.path.join(figures_folder, f"{clean_animal_name}_2D_plot_of_coordinates_with_highlighted_segments.png"))

        plt.close("all")
        fig, ax = plt.subplots(figsize=(23, 2))
        time_in_seconds = np.arange(0, max_analysis_time, 1 / frames_per_second)
        time_in_seconds = time_in_seconds[: len(centro_x.iloc[0 : max_analysis_time * frames_per_second])]

        plt.plot(time_in_seconds, centro_x.iloc[0 : max_analysis_time * frames_per_second], label="x coordinates", color="b", linewidth=2)
        for _, row in exploration_bouts_sec.iterrows():
            start_time = row["start (s)"]
            end_time = row["end (s)"]
            start_idx = int(start_time * frames_per_second)
            end_idx = int(end_time * frames_per_second)
            plt.plot(time_in_seconds[start_idx:end_idx], centro_x[start_idx:end_idx], color="r", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("X Coordinates")
        ax.set_title(f"X Coordinates with Highlighted Segments for Animal {clean_animal_name}")
        fig.tight_layout()

        fig.savefig(os.path.join(figures_folder, f"{clean_animal_name}_X_coordinates_with_highlighted_segments.png"))

        plt.close("all")
        fig, ax = plt.subplots(figsize=(23, 2))
        time_in_seconds = np.arange(0, max_analysis_time, 1 / frames_per_second)
        time_in_seconds = time_in_seconds[: len(centro_y.iloc[0 : max_analysis_time * frames_per_second])]

        plt.plot(time_in_seconds, centro_y.iloc[0 : max_analysis_time * frames_per_second], label="x coordinates", color="b", linewidth=2)
        for _, row in exploration_bouts_sec.iterrows():
            start_time = row["start (s)"]
            end_time = row["end (s)"]
            start_idx = int(start_time * frames_per_second)
            end_idx = int(end_time * frames_per_second)
            plt.plot(time_in_seconds[start_idx:end_idx], centro_y[start_idx:end_idx], color="r", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y Coordinates")
        ax.set_title(f"Y Coordinates with Highlighted Segments for Animal {clean_animal_name}")
        fig.tight_layout()

        fig.savefig(os.path.join(figures_folder, f"{clean_animal_name}_Y_coordinates_with_highlighted_segments.png"))
        plt.close("all")
        all_results[animal.name] = exploration_bouts_sec
        text_signal.emit((f"Processed {animal.name}...", "log_data_process_lineedit"))

    cleaned_results = {}
    for key in all_results.keys():
        try:
            # Replace the date and time timestamp with the an empty string
            # Replace all remaining underscores, spcaces and dashes with a single underscore
            new_key = re.sub(r"\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}", "", key)
            new_key = re.sub(r"[_\s-]+", "_", new_key)
            cleaned_results[new_key] = all_results[key]
        except AttributeError:
            print(f"Could not find the date and time timestamp for the animal {key}")
            print("Using the complete animal name instead.")
            cleaned_results[key] = key

    with pd.ExcelWriter(os.path.join(figures_folder, "Combined_results_by_sheet.xlsx"), engine="openpyxl") as writer:
        for animal_name, animal_df in cleaned_results.items():
            animal_df.to_excel(writer, sheet_name=animal_name, index=False)

    text_signal.emit(("clear_lineedit", "log_data_process_lineedit"))
    text_signal.emit(("All animals processed.", "log_data_process_lineedit"))
    self.options = {}


def find_sections(dataframe, framerate):
    in_interval = False
    start_idx = None
    intervals = pd.DataFrame(columns=["start", "end", "duration"])
    for idx, row in dataframe.iterrows():
        value = row[0]
        if value == 1 and not in_interval:
            start_idx = idx
            in_interval = True
        elif value == 0 and in_interval:
            duration = (idx - (start_idx - 1)) * (1 / framerate)
            new_dataframe = pd.DataFrame(data=[[start_idx, idx, duration]], columns=["start", "end", "duration"])
            if intervals.empty:
                intervals = new_dataframe
            else:
                intervals = pd.concat([intervals, new_dataframe])
            in_interval = False

    if in_interval:
        duration = (idx - (start_idx - 1)) * (1 / framerate)
        intervals = pd.concat([intervals, pd.DataFrame({"start": [start_idx], "end": [idx], "duration": [duration]})], ignore_index=True)
    return intervals


def is_inside_circle(x, y, roi_X, roi_Y, roi_D):
    # Calculate the radius
    radius = roi_D / 2.0

    # Calculate the distance between the point and the circle center
    distance = math.sqrt((x - roi_X) ** 2 + (y - roi_Y) ** 2)

    # Check if the distance is less than or equal to the radius
    return distance <= radius


def get_message():
    a = b"QmVoYXZ5dGhvbiBUb29sczogVGhlIHBsYWNlIHdoZXJlIHlvdXIgYnVncyB0aHJpdmUsV2VsY29tZSB0byBCZWhhdnl0aG9uIFRvb2xzOiBCZWNhdXNlIHRoZSBvbmx5IHdheSB0byBzdXJ2aXZlIGlzIHRvIGVtYnJhY2UgdGhlIGNoYW9zLixXZWxjb21lIHRvIEJlaGF2eXRob24gVG9vbHM6IFVubGVhc2hpbmcgY2hhb3Mgb25lIGJ1ZyBhdCBhIHRpbWUuLEJlaGF2eXRob24gVG9vbHM6IEJlY2F1c2Ugc3RhYmxlIGJ1aWxkcyBhcmUgb3ZlcnJhdGVkLixTdGVwIHJpZ2h0IHVwIHRvIEJlaGF2eXRob24gVG9vbHM6IFdoZXJlIGV2ZW4geW91ciBiZXN0IGNvZGUgY2FuIGdvIHdyb25nLixCZWhhdnl0aG9uIFRvb2xzOiBNYWtpbmcgcHJvZ3Jlc3MgZmVlbCBsaWtlIGEgZ2xpdGNoIGluIHRoZSBtYXRyaXguLEVtYnJhY2UgdGhlIHN0cnVnZ2xlIHdpdGggQmVoYXZ5dGhvbiBUb29sczogUGVyZmVjdGluZyBmcnVzdHJhdGlvbiB3aXRoIGVhY2ggdXBkYXRlLixCZWhhdnl0aG9uIFRvb2xzOiBXaGVyZSBjb2RpbmcgaXMgbW9yZSBhYm91dCBzdXJ2aXZhbCB0aGFuIHN1Y2Nlc3MuLEdldCBjb21mb3J0YWJsZSB3aXRoIEJlaGF2eXRob24gVG9vbHM6IEJlY2F1c2Ugbm90aGluZyB3b3JrcyBvbiB0aGUgZmlyc3QgdHJ5LixXZWxjb21lIHRvIEJlaGF2eXRob24gVG9vbHM6IFRoZSBhcnQgb2YgdHVybmluZyBlcnJvcnMgaW50byBuZXcgZmVhdHVyZXMuLEJlaGF2eXRob24gVG9vbHM6IFlvdXIgZGFpbHkgZG9zZSBvZiBkaWdpdGFsIG1hc29jaGlzbS4sUHJlcGFyZSBmb3IgQmVoYXZ5dGhvbiBUb29sczogV2hlcmUgdHJvdWJsZXNob290aW5nIHRha2VzIG9uIGEgbGlmZSBvZiBpdHMgb3duLixCZWhhdnl0aG9uIFRvb2xzOiBXaGVyZSBldmVuIHNpbXBsZSB0YXNrcyBiZWNvbWUgY29tcGxleCBwdXp6bGVzLFdlbGNvbWUgdG8gQmVoYXZ5dGhvbiBUb29sczogVGhlIHBsYXlncm91bmQgb2YgdW5leHBlY3RlZCBiZWhhdmlvcnMsQmVoYXZ5dGhvbiBUb29sczogV2hlcmUgZXZlcnkgbGluZSBvZiBjb2RlIGlzIGEgcG90ZW50aWFsIG1pbmVmaWVsZCxHZXQgcmVhZHkgZm9yIEJlaGF2eXRob24gVG9vbHM6IFlvdXIgYWR2ZW50dXJlIGluIG5ldmVyLWVuZGluZyBkZWJ1Z2dpbmcsQmVoYXZ5dGhvbiBUb29sczogQmVjYXVzZSBzYW5pdHkgaXMgb3ZlcnJhdGVkLFdlbGNvbWUgdG8gQmVoYXZ5dGhvbiBUb29sczogV2hlcmUgbG9naWMgZ29lcyB0byB0YWtlIGEgYnJlYWssQmVoYXZ5dGhvbiBUb29sczogVGhlIHNvZnR3YXJlIHRoYXQgdHVybnMgcm91dGluZSBpbnRvIGEgY2hhbGxlbmdlLEV4cGVjdCB0aGUgdW5leHBlY3RlZCB3aXRoIEJlaGF2eXRob24gVG9vbHM6IFdoZXJlIHN1cnByaXNlcyBhcmUgZ3VhcmFudGVlZCxCZWhhdnl0aG9uIFRvb2xzOiBUaGUgdWx0aW1hdGUgdGVzdCBvZiB5b3VyIHBhdGllbmNlIGFuZCBwZXJzaXN0ZW5jZSxTdXJ2aXZlIEJlaGF2eXRob24gVG9vbHM6IFdoZXJlIHRyaXVtcGggY29tZXMgYWZ0ZXIgY291bnRsZXNzIHJldHJpZXMsV2VsY29tZSB0byBCZWhhdnl0aG9uIFRvb2xzOiBBIGpvdXJuZXkgdGhyb3VnaCB0aGUgbGFuZCBvZiBnbGl0Y2hlcyxCZWhhdnl0aG9uIFRvb2xzOiBUcmFuc2Zvcm1pbmcgYnVncyBpbnRvIHVucGxhbm5lZCBmZWF0dXJlcyxHZXQgbG9zdCBpbiBCZWhhdnl0aG9uIFRvb2xzOiBXaGVyZSBjbGFyaXR5IGlzIGp1c3QgYW4gaWxsdXNpb24sQmVoYXZ5dGhvbiBUb29sczogV2hlcmUgdGhlIG9ubHkgY29uc3RhbnQgaXMgaW5jb25zaXN0ZW5jeSxXZWxjb21lIHRvIEJlaGF2eXRob24gVG9vbHM6IFRoZSBhcnQgb2YgbWFraW5nIHRoaW5ncyBtb3JlIGRpZmZpY3VsdCxCZWhhdnl0aG9uIFRvb2xzOiBZb3VyIGRhaWx5IHJlbWluZGVyIHRoYXQgbm90aGluZyBpcyBldmVyIHNpbXBsZSxOYXZpZ2F0ZSBjaGFvcyB3aXRoIEJlaGF2eXRob24gVG9vbHM6IFdoZXJlIG9yZGVyIGlzIGEgZGlzdGFudCBkcmVhbSxCZWhhdnl0aG9uIFRvb2xzOiBUaGUgYmVzdCBwbGFjZSB0byB0ZXN0IHlvdXIgd2lsbHBvd2VyLFdlbGNvbWUgdG8gQmVoYXZ5dGhvbiBUb29sczogV2hlcmUgZXhwZWN0YXRpb25zIGFuZCByZWFsaXR5IHJhcmVseSBtZWV0LEJlaGF2eXRob24gVG9vbHM6IFR1cm5pbmcgcm91dGluZSBjb2RpbmcgaW50byBhIHJvbGxlcmNvYXN0ZXIgcmlkZVdlbGNvbWUgdG8gQmVoYXZ5dGhvbiBUb29sczogd2hlcmUgeW91ciBtaXN0YWtlcyBiZWNvbWUgb3VyIGVudGVydGFpbm1lbnQsQ29uZ3JhdHVsYXRpb25zIG9uIGNob29zaW5nIEJlaGF2eXRob24gVG9vbHM6IHlvdXIgc2hvcnRjdXQgdG8gY29kZSBpbmR1Y2VkIGhlYWRhY2hlcyxCZWhhdnl0aG9uIFRvb2xzOiBCZWNhdXNlIGRlYnVnZ2luZyBpcyBmb3IgdGhlIHdlYWssRGl2ZSBpbnRvIEJlaGF2eXRob24gVG9vbHM6IHdoZXJlIHVzZXIgZnJpZW5kbHkgaXMganVzdCBhIG15dGgsV2VsY29tZSB0byBCZWhhdnl0aG9uIFRvb2xzOiBZb3VyIHBlcnNvbmFsIHRvdXIgb2YgcHJvZ3JhbW1pbmcgcHVyZ2F0b3J5LEJlaGF2eXRob24gVG9vbHM6IFBlcmZlY3QgZm9yIHRob3NlIHdobyBsb3ZlIHRoZSBzbWVsbCBvZiBmYWlsdXJlIGluIHRoZSBtb3JuaW5nLFN0YXJ0IHF1ZXN0aW9uaW5nIHlvdXIgbGlmZSBjaG9pY2VzOiBXZWxjb21lIHRvIEJlaGF2eXRob24gVG9vbHMsQmVoYXZ5dGhvbiBUb29sczogTWFraW5nIHNpbXBsZSB0YXNrcyBpbXBvc3NpYmx5IGNvbXBsaWNhdGVkIHNpbmNlIFllYXIgemVybyxFbmpveSBCZWhhdnl0aG9uIFRvb2xzOiB3ZSBwcm9taXNlIHlvdSB3aWxsIHJlZ3JldCBpdCxXZWxjb21lIHRvIEJlaGF2eXRob24gVG9vbHM6IFRoZSBwbGFjZSB3aGVyZSBidWdzIGZlZWwgYXQgaG9tZSxCZWhhdnl0aG9uIFRvb2xzOiBCZWNhdXNlIHdoYXQgaXMgbGlmZSB3aXRob3V0IGEgbGl0dGxlIHRvcnR1cmUsUHJlcGFyZSBmb3IgYSByaWRlIHRocm91Z2ggY2hhb3Mgd2l0aCBCZWhhdnl0aG9uIFRvb2xzLEJlaGF2eXRob24gVG9vbHM6IFdoZXJlIHNhbml0eSBnb2VzIHRvIGRpZSxXZWxjb21lIHRvIEJlaGF2eXRob24gVG9vbHM6IHRoZSBlcGl0b21lIG9mIGluZWZmaWNpZW5jeSxCZWhhdnl0aG9uIFRvb2xzOiBNYWtpbmcgc3VyZSB5b3UgbmV2ZXIgZ2V0IHRvbyBjb21mb3J0YWJsZSxTdGVwIHJpZ2h0IHVwIHRvIEJlaGF2eXRob24gVG9vbHM6IFlvdXIgZmFzdCB0cmFjayB0byBmcnVzdHJhdGlvbixCZWhhdnl0aG9uIFRvb2xzOiBUdXJuaW5nIGRyZWFtcyBpbnRvIG5pZ2h0bWFyZXMsV2VsY29tZSB0byBCZWhhdnl0aG9uIFRvb2xzOiB5b3VyIGRhaWx5IGRvc2Ugb2YgZGlnaXRhbCBkaXNhcHBvaW50bWVudCxCZWhhdnl0aG9uIFRvb2xzOiBXaGVuIHlvdSB3YW50IHRvIG1ha2UgeW91ciBwcm9ibGVtcyB3b3JzZSxXZWxjb21lIHRvIEJlaGF2eXRob24gVG9vbHM6IHdoZXJlIGV2ZXJ5IGZlYXR1cmUgaXMgYSBuZXcgZm9ybSBvZiBhZ29ueSxCZW0gdmluZG8gY29tcGFuaGVpcm8gZGUgZGlhcyBtYWxkaXRvcw=="
    sample = random.sample(base64.b64decode(a).decode().split(","), 1)[0]
    return sample
