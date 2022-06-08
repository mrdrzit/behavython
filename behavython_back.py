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
# from sklearn.neighbors import KernelDensity

# Alternative method using a function found here:
# https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work

# def kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs): 
#     """Build 2D kernel density estimate (KDE)."""
#     # create grid of sample locations (default: 100x100)
#     xx, yy = np.mgrid[x.min():x.max():xbins, 
#                       y.min():y.max():ybins]
#     xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
#     xy_train  = np.vstack([y, x]).T
#     kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
#     kde_skl.fit(xy_train)
#     # score_samples() returns the log-likelihood of the samples
#     z = np.exp(kde_skl.score_samples(xy_sample))
#     return xx, yy, np.reshape(z, xx.shape)

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
        arena_width = options['arena_width']                                    # Arena width seted by user 
        arena_height = options['arena_height']                                  # Arena height seted by user
        frames_per_second = options['frames_per_second']                        # Video frames per second seted by user
        threshold = options['threshold']                                        # Motion threshold seted by user in Bonsai
            
        video_height, video_width = self.last_frame.shape                       # Gets the video height and width from the video last frame
        factor_width = arena_width/video_width                                  # Calculates the width scale factor of the video
        factor_height = arena_height/video_height                               # Calculates the height scale factor of the video
        number_of_frames = len(self.data)                                       # Gets the number of frames
        
        x_axe = self.data[0]                                                    # Gets the x position 
        y_axe = self.data[1]                                                    # Gets the y position
        x_axe_cm = self.data[0]*factor_width                                    # Puts the x position on scale
        y_axe_cm = self.data[1]*factor_height                                   # Puts the y position on scale
        d_x_axe_cm = np.append(0, np.diff(self.data[0]))*factor_width           # Calculates the step difference of position in x axe
        d_y_axe_cm = np.append(0, np.diff(self.data[1]))*factor_height          # Calculates the step difference of position in y axe
                 
        displacement_raw = np.sqrt(np.square(d_x_axe_cm) + np.square(d_y_axe_cm))
        displacement = displacement_raw
        displacement[displacement < threshold] = 0
        
        accumulate_distance = np.cumsum(displacement)                           # Sums all the animal's movements and calculates the accumulated distance traveled
        total_distance = max(accumulate_distance)                               # Gets the animal's total distance traveled 
        
        time_vector = np.linspace(0, len(self.data)/frames_per_second, len(self.data))              # Creates a time vector
        velocity = np.divide(displacement, np.transpose(np.append(0, np.diff(time_vector))))        # Calculates the first derivate and finds the animal's velocity per time
        mean_velocity = np.nanmean(velocity)                                                        # Calculates the mean velocity from the velocity vector
        
        aceleration = np.divide(np.append(0, np.diff(velocity)), np.append(0, np.diff(time_vector)))    # Calculates the second derivate and finds the animal's acceleration per time
        movements = np.sum(displacement > 0)                                                            # Calculates the number of movements made by the animal
        time_moving = np.sum(displacement > 0)*(1/frames_per_second)                                    # Calculates the total time of movements made by the animal
        time_resting = np.sum(displacement == 0)*(1/frames_per_second)                                  # Calculates the total time of the animal without movimentations 

        kde_space_coordinates = np.array([np.array(x_axe), np.array(y_axe)])
        kde_instance = stats.gaussian_kde(kde_space_coordinates)
        point_density_function = kde_instance.evaluate(kde_space_coordinates)
        color_limits = np.array([(x - np.min(point_density_function))/(np.max(point_density_function) - np.min(point_density_function)) for x in point_density_function])
        
        # TODO create check to verify that the csv file have the correct number of data columns  
        quadrant_data = np.array(self.data[[2,3,4,5,6]])                                           # Extract the quadrant data from csv file
        colDif = np.abs(quadrant_data[:,0] - np.sum(quadrant_data[:][:,1:],axis=1))                # Here, the values will be off-by-one because MATLAB starts at 1
        full_entry_indexes = colDif != 1                                                           # Create a logical array where there is "full entry"
        time_spent = np.delete(quadrant_data, full_entry_indexes, 0)                               # True crossings over time (full crossings only) 
        quadrant_crossings = abs(np.diff(time_spent, axis=0))
        total_time_in_quadrant = np.sum(np.divide(time_spent,frames_per_second),0)                  # Total time spent in each quadrant
        total_number_of_entries = np.sum(quadrant_crossings > 0, 0)                                 # Total # of entries in each quadrant

        # Alternative method using a function found here:
        # https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
        # xx, yy, zz = kde2D(x, y, 1.0, 50j, 100j)
        # fig1 = plt.figure()
        # plt.pcolormesh(xx, yy, zz)
        # plt.scatter(x, y, s=0.01, facecolor='white')
        # plt.show()
        
        self.analyse_results = {'video_width'         : video_width,
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
        
        return self.analyse_results    

    def plot_analyse(self):
        # Figure 1 - 
        figure_0, axe_0 = plt.subplots()
        movement_points = np.array([self.analyse_results["x_axe"], self.analyse_results["y_axe"]]).T.reshape(-1, 1, 2) 
        movement_segments = np.concatenate([movement_points[:-1], movement_points[1:]], axis=1)                         # Creates a 2D array containing the line segments coordinates
        movement_line_collection = LineCollection(movement_segments, cmap="CMRmap", linewidth=1.5)                      # TODO edit this line to customize the movement graph
        movement_line_collection.set_array(self.analyse_results["color_limits"])                                                                # Set the line color to the normalized values of "color_limits"
        axe_0.add_collection(movement_line_collection)
        axe_0.autoscale_view()
        plt.show()
        
        # Figure 1 - 
        #plt.rcParams["figure.figsize"] = [7.00, 3.50]
        #plt.rcParams["figure.autolayout"] = True
        # im = plt.imread(self.directory + ".png")
        # figure_1, axe_1 = plt.subplots()
        # im = axe_1.imshow(im, extent=[0, self.analyse_results['video_width'], 0, self.analyse_results['video_height']])
        # axe_1.add_collection(movement_line_collection)
        # axe_1.autoscale_view()
        # plt.show()
        
        # Figure 2 - Histogram
        figure_2, axe_2 = plt.subplots()
        axe_2.hist(self.analyse_results['displacement'], 400, density=True, facecolor='g', alpha=0.75)
        
        # Figure 3 - Time spent on each arm over time
        figure_3, ((axe_11, axe_12, axe_13), (axe_21, axe_22, axe_23), (axe_31, axe_32, axe_33)) = plt.subplots(3,3)
        figure_3.delaxes(axe_11)
        figure_3.delaxes(axe_13)
        figure_3.delaxes(axe_31)
        figure_3.delaxes(axe_33)
          
        axe_12.plot(self.analyse_results["time_spent"][:,0])
        axe_12.set_ylim((0, 1.5))
        axe_12.set_title('upper arm')
    
        axe_21.plot(self.analyse_results["time_spent"][:,1])
        axe_21.set_ylim((0, 1.5))
        axe_21.set_title('left  arm')
    
        axe_22.plot(self.analyse_results["time_spent"][:,2])
        axe_22.set_ylim((0, 1.5))
        axe_22.set_title('center')
    
        axe_23.plot(self.analyse_results["time_spent"][:,3])
        axe_23.set_ylim((0, 1.5))
        axe_23.set_title('right arm')
    
        axe_32.plot(self.analyse_results["time_spent"][:,4])
        axe_32.set_ylim((0, 1.5))
        axe_32.set_title('lower arm')
        
        figure_3.suptitle('Time spent on each arm over time')
        plt.tight_layout()
        plt.show()
        
        # Figure 4 - Number of crossings
        figure_4, ((axe_11, axe_12, axe_13), (axe_21, axe_22, axe_23), (axe_31, axe_32, axe_33)) = plt.subplots(3,3)
        figure_4.delaxes(axe_11)
        figure_4.delaxes(axe_13)
        figure_4.delaxes(axe_31)
        figure_4.delaxes(axe_33)
          
        axe_12.plot(self.analyse_results["quadrant_crossings"][:,0])
        axe_12.set_ylim((0, 1.5))
        axe_12.set_title('upper arm')
    
        axe_21.plot(self.analyse_results["quadrant_crossings"][:,1])
        axe_21.set_ylim((0, 1.5))
        axe_21.set_title('left  arm')
    
        axe_22.plot(self.analyse_results["quadrant_crossings"][:,2])
        axe_22.set_ylim((0, 1.5))
        axe_22.set_title('center')
    
        axe_23.plot(self.analyse_results["quadrant_crossings"][:,3])
        axe_23.set_ylim((0, 1.5))
        axe_23.set_title('right arm')
    
        axe_32.plot(self.analyse_results["quadrant_crossings"][:,4])
        axe_32.set_ylim((0, 1.5))
        axe_32.set_title('lower arm')
        
        figure_4.suptitle('Number of crossings')
        plt.tight_layout()
        plt.show()
    
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
    def get_experiments(self):
        file_explorer = tk.Tk()
        file_explorer.withdraw()
        file_explorer.call('wm', 'attributes', '.', '-topmost', True)
        selected_files = filedialog.askopenfilename(title = "Select the files", multiple = True) 
    
        files = files_class()
        files.add_files(selected_files)
    
        experiments = []
    
        for index in range(0, files.number):
            
            experiments.append(experiment_class())
            
            experiments[index].name = files.name[index]  
            experiments[index].directory = files.directory[index]
            try:
                raw_data = pd.read_csv(files.directory[index] + '.csv', sep = ',', na_values = ['no info', '.'], header = None)
                experiments[index].data = raw_data.interpolate(method='spline', order=1, limit_direction = 'both', axis = 0)
                print("Trying to read file " + experiments[index].name + '.csv')
            except:
                print("Não existe arquivo CSV com o nome " + files.name[index])
            try:
                experiments[index].last_frame = rgb2gray(skimage.io.imread(files.directory[index] + '.png'))
                print("Trying to read file " + experiments[index].name + '.png')
            except:
                print("Não existe arquivo PNG com o nome " + files.name[index])
                
        return experiments
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    