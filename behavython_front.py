import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import os
import skimage.io
import tkinter as tk
import pandas as pd
import numpy as np
from matplotlib.collections import LineCollection 
from matplotlib import pyplot as plt 
from tkinter import filedialog
from skimage.color import rgb2gray
from scipy import stats

class analysis_class(QObject):
    ''' 

    '''
    finished = pyqtSignal()                                                                         # Signal that will be output to the interface when the function is completed
    progress_bar = pyqtSignal(int)
    
    def __init__(self, experiments, options, plot_viewer):                                          # Initializes when the thread is started
        ''' 
        This private function is executed when the class is called, and all parameters are
        defined here
        '''
        super(analysis_class, self).__init__()                                                      # Super declaration
        self.experiments = experiments                                                              # Sets the the interface plot widget as self variable
        self.options = options
        self.plot_viewer = plot_viewer
        
    def run_analyse(self):
        for i in range(0, len(self.experiments)):
            analyse_results, data_frame = self.experiments[i].video_analyse(self.options)
            if i == 0:
                results_data_frame = data_frame
            else:
                results_data_frame = results_data_frame.join(data_frame, how='outer')

            if self.options['experiment_type'] == 'open_field':
                self.experiments[i].plot_analysis_open_field(self.plot_viewer, i)
            else:
                self.experiments[i].plot_analysis_pluz_maze(self.plot_viewer, i)
            self.progress_bar.emit(round(((i+1)/len(self.experiments))*100))
        
        results_data_frame.to_excel(self.experiments[0].directory + '_rusults.xlsx')
        true_path = os.path.dirname(__file__) + '\\Video_analyse_validation\\animal_2_PYTHON.xlsx'
        results_data_frame.to_excel(true_path, header=False)
        self.finished.emit()

class behavython_gui(QMainWindow):
    '''
    This class contains all the commands of the interface as well as the constructors of the 
    interface itself.
    '''
    def __init__(self):
        '''
        This private function calls the interface of a .ui file created in Qt Designer.
        '''

        super(behavython_gui, self).__init__()                                      # Calls the inherited classes __init__ method
        load_gui_path = os.path.dirname(__file__) + "\\Behavython_GUI.ui"
        uic.loadUi(load_gui_path, self, package='behavython_front')                 # Loads the interface design archive (made in Qt Designer)
        self.show()

        self.options = {}
        
        self.clear_button.clicked.connect(self.clear_function)        
        self.analysis_button.clicked.connect(self.analysis_function)
        
    def analysis_function(self):
        plot_viewer_function()
        self.resume_lineedit.clear()
        
        if self.type_combobox.currentIndex() == 1:
            self.options['experiment_type'] = 'open_field'
        else:
            self.options['experiment_type'] = 'plus_maze'
        self.options['arena_width'] = int(self.arena_width_lineedit.text())
        self.options['arena_height'] = int(self.arena_height_lineedit.text())
        self.options['frames_per_second'] = float(self.frames_per_second_lineedit.text())
        if self.animal_combobox.currentIndex() == 0:
            self.options['threshold'] = 0.0267
        else:
            self.options['threshold'] = 0.0267
        
        functions = interface_functions()
        self.experiments = functions.get_experiments(self.resume_lineedit, self.options['experiment_type'])
            
        self.analysis_thread = QThread()                                                          # Creates a QThread object to plot the received data
        self.analysis_worker = analysis_class(self.experiments, self.options, self.plot_viewer)   # Creates a worker object named plot_data_class
        self.analysis_worker.moveToThread(self.analysis_thread)                                   # Moves the class to the thread
        self.analysis_worker.finished.connect(self.analysis_thread.quit)                          # When the process is finished, this command quits the worker
        self.analysis_worker.finished.connect(self.analysis_thread.wait)                          # When the process is finished, this command waits the worker to finish completely
        self.analysis_worker.finished.connect(self.analysis_worker.deleteLater)                   # When the process is finished, this command deletes the worker
        self.analysis_worker.progress_bar.connect(self.progress_bar_function)
        self.analysis_thread.finished.connect(self.analysis_thread.deleteLater)                   # When the process is finished, this command deletes the thread.
        self.analysis_thread.start()                                                              # Starts the thread 

        self.analysis_worker.run_analyse()
    
    def progress_bar_function(self, value):
        self.progress_bar.setValue(value)
                
    def clear_function(self):
        self.options = {}
        self.type_combobox.setCurrentIndex(0)
        self.frames_per_second_lineedit.setText('30')
        self.arena_width_lineedit.setText('65')
        self.arena_height_lineedit.setText('65')
        self.animal_combobox.setCurrentIndex(0)
        plot_viewer_function()

class plot_viewer_function(QtWidgets.QWidget):
    '''
    This class modifies the interface's QWidget in order to insert a plot viewer.
    
    TODO: Find the best way to plot a summary of the data.
    E.g.: A button to advance or go back in the plots
    '''    
    def __init__(self, parent = None):
        QtWidgets.QWidget.__init__(self, parent)                                   
        self.canvas = FigureCanvas(Figure(facecolor = "#353535", dpi=100, tight_layout=True))          # Create a figure object 
        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, 
                                           QtWidgets.QSizePolicy.Policy.Expanding)  # Creates axes size policy
        self.canvas.setSizePolicy(sizePolicy)                                       # Sets the size policy
        
        self.canvas.axes = []
        for i in range(1,10):
            self.canvas.axes.append(self.canvas.figure.add_subplot(3,3,i))          # Creates an empty plot
            self.canvas.axes[i-1].set_facecolor("#252525")                          # Changes the plot face color
            self.canvas.axes[i-1].get_xaxis().set_visible(False)
            self.canvas.axes[i-1].get_yaxis().set_visible(False)
        
        vertical_layout = QtWidgets.QVBoxLayout()                                   # Creates a layout
        vertical_layout.addWidget(self.canvas)                                      # Inserts the figure on the layout
        self.canvas.figure.subplots_adjust(left=0, bottom=0, right=1, 
                                           top=1, wspace=0, hspace=0)               # Sets the plot margins 
        self.setLayout(vertical_layout)                                             # Sets the layout

class experiment_class:
    '''
    This class declares each experiment and makes the necessary calculations to 
    analyze the movement of the animal in the experimental box. 
    
    For each experiment one object of this class will be created and added in
    a experiments list. 
    '''
    
    def __init__(self):
        self.name = []                                                          # Experiment name
        self.data = []                                                          # Experiment loaded data
        self.last_frame = []                                                    # Experiment last behavior video frame
        self.directory = []                                                     # Experiment directory                    
        
    def video_analyse(self, options):       
        self.experiment_type = options['experiment_type']
        arena_width = options['arena_width']                                    # Arena width set by user 
        arena_height = options['arena_height']                                  # Arena height set by user
        frames_per_second = options['frames_per_second']                        # Video frames per second set by user
        threshold = options['threshold']                                        # Motion threshold set by user in Bonsai
            
        video_height, video_width = self.last_frame.shape                       # Gets the video height and width from the video's last frame
        factor_width = arena_width/video_width                                  # Calculates the width scale factor of the video
        factor_height = arena_height/video_height                               # Calculates the height scale factor of the video
        number_of_frames = len(self.data)                                       # Gets the number of frames
        
        x_axe = self.data[0]                                                    # Gets the x position 
        y_axe = self.data[1]                                                    # Gets the y position
        x_axe_cm = self.data[0]*factor_width                                    # Puts the x position on scale
        y_axe_cm = self.data[1]*factor_height                                   # Puts the y position on scale
        d_x_axe_cm = np.append(0, np.diff(self.data[0]))*factor_width           # Calculates the step difference of position in x axis
        d_y_axe_cm = np.append(0, np.diff(self.data[1]))*factor_height          # Calculates the step difference of position in y axis
                 
        displacement_raw = np.sqrt(np.square(d_x_axe_cm) + np.square(d_y_axe_cm))
        displacement = displacement_raw
        displacement[displacement < threshold] = 0
        
        accumulate_distance = np.cumsum(displacement)                           # Sums all the animal's movements and calculates the accumulated distance traveled
        total_distance = max(accumulate_distance)                               # Gets the animal's total distance traveled 
        
        time_vector = np.linspace(0, len(self.data)/frames_per_second, len(self.data))              # Creates a time vector
        velocity = np.divide(displacement, np.transpose(np.append(0, np.diff(time_vector))))        # Calculates the first derivate and finds the animal's velocity per time
        mean_velocity = np.nanmean(velocity)                                                        # Calculates the mean velocity from the velocity vector
        
        aceleration = np.divide(np.append(0, np.diff(velocity)), np.append(0, np.diff(time_vector)))    # Calculates the second derivative and finds the animal's acceleration per time
        movements = np.sum(displacement > 0)                                                            # Calculates the number of movements made by the animal
        time_moving = np.sum(displacement > 0)*(1/frames_per_second)                                    # Calculates the total time of movements made by the animal
        time_resting = np.sum(displacement == 0)*(1/frames_per_second)                                  # Calculates the total time of the animal without movimentations 

        kde_space_coordinates = np.array([np.array(x_axe), np.array(y_axe)])
        kde_instance = stats.gaussian_kde(kde_space_coordinates)
        point_density_function = kde_instance.evaluate(kde_space_coordinates)
        color_limits = np.array([(x - np.min(point_density_function))/(np.max(point_density_function) - np.min(point_density_function)) for x in point_density_function])
          
        quadrant_data = np.array(self.data[self.data.columns[2:]])                                  # Extract the quadrant data from csv file
        colDif = np.abs(quadrant_data[:,0] - np.sum(quadrant_data[:][:,1:],axis=1))                 # Here, the values will be off-by-one because MATLAB starts at 1
        full_entry_indexes = colDif != 1                                                            # Create a logical array where there is "full entry"
        time_spent = np.delete(quadrant_data, full_entry_indexes, 0)                                # True crossings over time (full crossings only) 
        quadrant_crossings = abs(np.diff(time_spent, axis=0))
        total_time_in_quadrant = np.sum(np.divide(time_spent,frames_per_second),0)                  # Total time spent in each quadrant
        total_number_of_entries = np.sum(quadrant_crossings > 0, 0)                                 # Total # of entries in each quadrant
        
        self.analysis_results = {'video_width'         : video_width,
                                 'video_height'        : video_height,
                                 'number_of_frames'    : number_of_frames,
                                 'x_axe'               : x_axe,
                                 'y_axe'               : y_axe,
                                 'x_axe_cm'            : x_axe_cm,
                                 'y_axe_cm'            : y_axe_cm,
                                 'd_x_axe_cm'          : d_x_axe_cm,
                                 'd_y_axe_cm'          : d_y_axe_cm,
                                 'displacement'        : displacement,
                                 'accumulate_distance' : accumulate_distance,
                                 'total_distance'      : total_distance,
                                 'time_vector'         : time_vector,
                                 'velocity'            : velocity,
                                 'mean_velocity'       : mean_velocity,
                                 'aceleration'         : aceleration,
                                 'movements'           : movements,
                                 'time_spent'          : time_spent,
                                 'time_moving'         : time_moving,
                                 'time_resting'        : time_resting,
                                 'quadrant_crossings'  : quadrant_crossings,
                                 'time_in_quadrant'    : total_time_in_quadrant,
                                 'number_of_entries'   : total_number_of_entries,
                                 'color_limits'        : color_limits
                                 }
        
        
        # print(self.analysis_results)
        
        dict_to_excel = {'Total distance (cm)'          : total_distance,
                         'Mean velocity (cm/s)'         : mean_velocity,
                         'Movements'                    : movements,
                         'Time moving (s)'              : time_moving,
                         'Time resting(s)'              : time_resting,
                         'Total time at upper arm (s)'  : total_time_in_quadrant[0],
                         'Total time at lower arm (s)'  : total_time_in_quadrant[4],
                         'Total time at left arm (s)'   : total_time_in_quadrant[1],
                         'Total time at right arm (s)'  : total_time_in_quadrant[3],
                         'Total time at center (s)'     : total_time_in_quadrant[2],
                         'Crossings to the upper arm'   : total_number_of_entries[0],
                         'Crossings to the lower arm'   : total_number_of_entries[4],
                         'Crossings to the left arm'    : total_number_of_entries[1],
                         'Crossings to the right arm'   : total_number_of_entries[3],
                         'Crossings to the center'      : total_number_of_entries[2],
                         }
        
        data_frame = pd.DataFrame(data = dict_to_excel, index=[self.name])
        data_frame = (data_frame.T)
        return self.analysis_results, data_frame    

    def plot_analysis_pluz_maze(self, plot_viewer, plot_number):
        # Figure 1 - 
        movement_points = np.array([self.analysis_results["x_axe"], self.analysis_results["y_axe"]]).T.reshape(-1, 1, 2) 
        movement_segments = np.concatenate([movement_points[:-1], movement_points[1:]], axis=1)                         # Creates a 2D array containing the line segments coordinates
        movement_line_collection = LineCollection(movement_segments, cmap="CMRmap", linewidth=1.5)                      # Creates a LineCollection object with custom color map
        movement_line_collection.set_array(self.analysis_results["color_limits"])                                       # Set the line color to the normalized values of "color_limits"
        figure_1, axe_1 = plt.subplots()
        #plt.rcParams["figure.figsize"] = [7.00, 3.50]
        #plt.rcParams["figure.autolayout"] = True
        im = plt.imread(self.directory + ".png")
        axe_1.imshow(im)
        axe_1.add_collection(movement_line_collection)
        axe_1.axis('tight')
        axe_1.axis('off')
        figure_1.subplots_adjust(left=0,right=1,bottom=0,top=1)
        plt.savefig(self.directory + '_1.png', frameon='false')
        plt.autoscale()
        plt.show()
        plt.close(figure_1)
        
        im = plt.imread("temp.png")
        plot_viewer.canvas.axes[plot_number].imshow(im)
        #plot_viewer.canvas.axes[plot_number].title('Experiment ' + str(plot_number+1), fontsize = 10, fontfamily="DejaVu Sans", color="white")
        plot_viewer.canvas.draw_idle()
        
                
        # Figure 2 - Histogram
        #figure_2, axe_2 = plt.subplots()
        #axe_2.hist(self.analysis_results['displacement'], 400, density=True, facecolor='g', alpha=0.75)
        #plt.show()
        # plt.savefig(self.directory + '_2.png')
        # plt.close(figure_2)
        
        # Figure 3 - Time spent on each arm over time
        figure_3, ((axe_11, axe_12, axe_13), (axe_21, axe_22, axe_23), (axe_31, axe_32, axe_33)) = plt.subplots(3,3)
        figure_3.delaxes(axe_11)
        figure_3.delaxes(axe_13)
        figure_3.delaxes(axe_31)
        figure_3.delaxes(axe_33)
        
        axe_12.plot(self.analysis_results["time_spent"][:,0], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,0]) == 1
        axe_12.plot(self.analysis_results["quadrant_crossings"][:,0], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_12.set_ylim((0, 1.5))
        axe_12.set_title('upper arm')
    
        axe_21.plot(self.analysis_results["time_spent"][:,1], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,1]) == 1
        axe_21.plot(self.analysis_results["quadrant_crossings"][:,1], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_21.set_ylim((0, 1.5))
        axe_21.set_title('left  arm')
    
        axe_22.plot(self.analysis_results["time_spent"][:,2], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,2]) == 1
        axe_22.plot(self.analysis_results["quadrant_crossings"][:,2], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_22.set_ylim((0, 1.5))
        axe_22.set_title('center')
    
        axe_23.plot(self.analysis_results["time_spent"][:,3], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,3]) == 1
        axe_23.plot(self.analysis_results["quadrant_crossings"][:,3], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_23.set_ylim((0, 1.5))
        axe_23.set_title('right arm')
    
        axe_32.plot(self.analysis_results["time_spent"][:,4], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,4]) == 1
        axe_32.plot(self.analysis_results["quadrant_crossings"][:,4], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_32.set_ylim((0, 1.5))
        axe_32.set_title('lower arm')
        
        figure_3.suptitle('Time spent on each arm over time')
        plt.tight_layout()
        plt.show()
        # plt.savefig(self.directory + '_3.png')
        plt.close(figure_3)
        
        # Figure 4 - Number of crossings
        # figure_4, ((axe_11, axe_12, axe_13), (axe_21, axe_22, axe_23), (axe_31, axe_32, axe_33)) = plt.subplots(3,3)
        # figure_4.delaxes(axe_11)
        # figure_4.delaxes(axe_13)
        # figure_4.delaxes(axe_31)
        # figure_4.delaxes(axe_33)
            
        # axe_12.plot(self.analysis_results["quadrant_crossings"][:,0])
        # axe_12.set_ylim((0, 1.5))
        # axe_12.set_title('upper arm')
    
        # axe_21.plot(self.analysis_results["quadrant_crossings"][:,1])
        # axe_21.set_ylim((0, 1.5))
        # axe_21.set_title('left  arm')
    
        # axe_22.plot(self.analysis_results["quadrant_crossings"][:,2])
        # axe_22.set_ylim((0, 1.5))
        # axe_22.set_title('center')
    
        # axe_23.plot(self.analysis_results["quadrant_crossings"][:,3])
        # axe_23.set_ylim((0, 1.5))
        # axe_23.set_title('right arm')
    
        # axe_32.plot(self.analysis_results["quadrant_crossings"][:,4])
        # axe_32.set_ylim((0, 1.5))
        # axe_32.set_title('lower arm')
        
        # figure_4.suptitle('Number of crossings')
        # plt.tight_layout()
        # plt.show()            

def plot_analysis_open_field(self, plot_viewer, plot_number):
        # Figure 1 - 
        figure_0, axe_0 = plt.subplots()
        movement_points = np.array([self.analysis_results["x_axe"], self.analysis_results["y_axe"]]).T.reshape(-1, 1, 2) 
        movement_segments = np.concatenate([movement_points[:-1], movement_points[1:]], axis=1)                         # Creates a 2D array containing the line segments coordinates
        movement_line_collection = LineCollection(movement_segments, cmap="CMRmap", linewidth=1.5)                      # Creates a LineCollection object with custom color map
        movement_line_collection.set_array(self.analysis_results["color_limits"])                                       # Set the line color to the normalized values of "color_limits"
        axe_0.add_collection(movement_line_collection)
        axe_0.autoscale_view()
        plt.show()
        # plt.savefig(self.directory + '_0.png')
        # plt.close(figure_0)
        
        plot_viewer.canvas.axes[plot_number].plot(self.analysis_results["x_axe"], self.analysis_results["y_axe"])
        #plot_viewer.canvas.axes[plot_number].title('Experiment ' + str(plot_number+1), fontsize = 10, fontfamily="DejaVu Sans", color="white")
        plot_viewer.canvas.draw_idle()
        
        # Figure 1 - 
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        im = plt.imread(self.directory + ".png")
        figure_1, axe_1 = plt.subplots()
        im = axe_1.imshow(im)
        axe_1.add_collection(movement_line_collection)
        plt.autoscale()
        plt.show()
        # plt.savefig(self.directory + '_1.png')
        # plt.close(figure_1)
        
        # Figure 2 - Histogram
        figure_2, axe_2 = plt.subplots()
        axe_2.hist(self.analysis_results['displacement'], 400, density=True, facecolor='g', alpha=0.75)
        plt.show(figure_2)
        # plt.savefig(self.directory + '_2.png')
        # plt.close(figure_2)
        
        # Figure 3 - Time spent on each arm over time
        figure_3, ((axe_11, axe_12, axe_13), (axe_21, axe_22, axe_23), (axe_31, axe_32, axe_33)) = plt.subplots(3,3)
        figure_3.delaxes(axe_11)
        figure_3.delaxes(axe_13)
        figure_3.delaxes(axe_31)
        figure_3.delaxes(axe_33)
        
        axe_12.plot(self.analysis_results["time_spent"][:,0], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,0]) == 1
        axe_12.plot(self.analysis_results["quadrant_crossings"][:,0], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_12.set_ylim((0, 1.5))
        axe_12.set_title('upper arm')
    
        axe_21.plot(self.analysis_results["time_spent"][:,1], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,1]) == 1
        axe_21.plot(self.analysis_results["quadrant_crossings"][:,1], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_21.set_ylim((0, 1.5))
        axe_21.set_title('left  arm')
    
        axe_22.plot(self.analysis_results["time_spent"][:,2], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,2]) == 1
        axe_22.plot(self.analysis_results["quadrant_crossings"][:,2], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_22.set_ylim((0, 1.5))
        axe_22.set_title('center')
    
        axe_23.plot(self.analysis_results["time_spent"][:,3], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,3]) == 1
        axe_23.plot(self.analysis_results["quadrant_crossings"][:,3], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_23.set_ylim((0, 1.5))
        axe_23.set_title('right arm')
    
        axe_32.plot(self.analysis_results["time_spent"][:,4], color = '#2C53A1')
        entries = np.array(self.analysis_results["quadrant_crossings"][:,4]) == 1
        axe_32.plot(self.analysis_results["quadrant_crossings"][:,4], 'o', ms = 2, markevery=entries, markerfacecolor='#A21F27', markeredgecolor='#A21F27')
        axe_32.set_ylim((0, 1.5))
        axe_32.set_title('lower arm')
        
        figure_3.suptitle('Time spent on each arm over time')
        plt.tight_layout()
        plt.show()
        # plt.savefig(self.directory + '_3.png')
        # plt.close(figure_3)
        
        # Figure 4 - Number of crossings
        # figure_4, ((axe_11, axe_12, axe_13), (axe_21, axe_22, axe_23), (axe_31, axe_32, axe_33)) = plt.subplots(3,3)
        # figure_4.delaxes(axe_11)
        # figure_4.delaxes(axe_13)
        # figure_4.delaxes(axe_31)
        # figure_4.delaxes(axe_33)
            
        # axe_12.plot(self.analysis_results["quadrant_crossings"][:,0])
        # axe_12.set_ylim((0, 1.5))
        # axe_12.set_title('upper arm')
    
        # axe_21.plot(self.analysis_results["quadrant_crossings"][:,1])
        # axe_21.set_ylim((0, 1.5))
        # axe_21.set_title('left  arm')
    
        # axe_22.plot(self.analysis_results["quadrant_crossings"][:,2])
        # axe_22.set_ylim((0, 1.5))
        # axe_22.set_title('center')
    
        # axe_23.plot(self.analysis_results["quadrant_crossings"][:,3])
        # axe_23.set_ylim((0, 1.5))
        # axe_23.set_title('right arm')
    
        # axe_32.plot(self.analysis_results["quadrant_crossings"][:,4])
        # axe_32.set_ylim((0, 1.5))
        # axe_32.set_title('lower arm')
        
        # figure_4.suptitle('Number of crossings')
        # plt.tight_layout()
        # plt.show()

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
    def get_experiments(self, line_edit, experiment_type):
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call('wm', 'attributes', '.', '-topmost', True)
        selected_files = filedialog.askopenfilename(title = "Select the files", multiple = True) 
    
        files = files_class()
        files.add_files(selected_files)
    
        experiments = []
    
        for index in range(0, files.number):
            
            experiments.append(experiment_class())
            
            try:
                raw_data = pd.read_csv(files.directory[index] + '.csv', sep = ',', na_values = ['no info', '.'], header = None)
                raw_image = rgb2gray(skimage.io.imread(files.directory[index] + '.png'))
            except:
                line_edit.append("- Doesn't exists CSV or PNG file with name " + files.name[index])
                experiments.pop()
            else:
                if raw_data.shape[1] == 7 and experiment_type == 'plus_maze':
                    experiments[index].data = raw_data.interpolate(method='spline', order=1, limit_direction = 'both', axis = 0)
                    line_edit.append("- File " + files.name[index] + '.csv was read')
                    experiments[index].last_frame = raw_image
                    line_edit.append("- File " + files.name[index] + '.png was read')
                    experiments[index].name = files.name[index]  
                    experiments[index].directory = files.directory[index]
                elif experiment_type == 'open_field':
                    experiments[index].data = raw_data.interpolate(method='spline', order=1, limit_direction = 'both', axis = 0)
                    line_edit.append("- File " + files.name[index] + '.csv was read')
                    experiments[index].last_frame = raw_image
                    line_edit.append("- File " + files.name[index] + '.png was read')
                    experiments[index].name = files.name[index]  
                    experiments[index].directory = files.directory[index]
                else:
                    line_edit.append("- The " + files.name[index] + ".csv file had more columns than the elevated plus maze test allows")
                    
        return experiments
 



def main(): 
  app = QtWidgets.QApplication(sys.argv)   # Create an instance of QtWidgets.QApplication
  window = behavython_gui()                # Create an instance of our class
  app.exec_()                              # Start the application
 
if __name__ == '__main__': 
  show = main()


# if __name__ == '__main__':   
#     def main():
#         if not QtWidgets.QApplication.instance():
#             QtWidgets.QApplication(sys.argv)
#         else:
#             QtWidgets.QApplication.instance()
#             main = behavython_gui()
#             main.show()
#             return main  
#     show = main()