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
    def __init__(self):
        self.name = []
        self.data = []
        self.last_frame = []
             
    def video_analyse(self, options):       
        arena_width = options['arena_width']
        arena_height = options['arena_height']
        frames_per_second = options['frames_per_second']
        threshold = options['threshold']
            
        video_height, video_width = self.last_frame.shape
        factor_width = arena_width/video_width
        factor_height = arena_height/video_height
        number_of_frames = len(self.data)
        
        x_axe = self.data[0]
        y_axe = self.data[1] 
        x_axe_cm = self.data[0]*factor_width
        y_axe_cm = self.data[1]*factor_height
        d_x_axe_cm = np.append(0, np.diff(self.data[0]))*factor_width
        d_y_axe_cm = np.append(0, np.diff(self.data[1]))*factor_height
                 
        displacement_raw = np.sqrt(np.square(d_x_axe_cm) + np.square(d_y_axe_cm))
        displacement = displacement_raw
        displacement[displacement < threshold] = 0
        
        accumulate_distance = np.cumsum(displacement)
        total_distance = max(accumulate_distance)
        
        time_vector = np.linspace(0, len(self.data)/frames_per_second, len(self.data))
        velocity = np.divide(displacement, np.transpose(np.append(0, np.diff(time_vector))))        # Expand steps to make the code more readable
        mean_velocity = np.nanmean(velocity)
        
        aceleration = np.divide(np.append(0, np.diff(velocity)), np.append(0, np.diff(time_vector)))
        moviments = np.sum(displacement > 0)
        time_moviments = np.sum(displacement > 0)*(1/frames_per_second)
        time_resting = np.sum(displacement == 0)*(1/frames_per_second)

        kde_space_coordinates = np.array([np.array(x_axe), np.array(y_axe)])
        kde_instance = stats.gaussian_kde(kde_space_coordinates)
        point_density_function = kde_instance.evaluate(kde_space_coordinates)
        color_limits = np.array([(x - np.min(point_density_function))/(np.max(point_density_function) - np.min(point_density_function)) for x in point_density_function])
        
        fig, ax = plt.subplots()
        movement_points = np.array([x_axe, y_axe]).T.reshape(-1, 1, 2) 
        movement_segments = np.concatenate([movement_points[:-1], movement_points[1:]], axis=1)      # Creates a 2D array containing the line segments coordinates
        movement_line_collection = LineCollection(movement_segments, cmap="CMRmap", linewidth=1.5)   # TODO edit this line to customize the movement graph
        movement_line_collection.set_array(color_limits)                                             # Set the line color to the normalized values of "color_limits"
        ax.add_collection(movement_line_collection)
        ax.autoscale_view()
        # plt.show()

        # TODO create check to verify that the csv file have the correct number of data columns  
        quadrant_data = np.array(self.data[[2,3,4,5,6]])                                           # Extract the quadrant data from csv file
        colDif = np.abs(quadrant_data[:,0] - np.sum(quadrant_data[:][:,1:],axis=1))                # Here, the values will be off-by-one because MATLAB starts at 1
        full_entry_indexes = colDif != 1                                                           # Create a logical array where there is "full entry"
        timespent = np.delete(quadrant_data, full_entry_indexes, 0)                                # True crossings over time (full crossings only) 
        number_of_quadrant_crossing = abs(np.diff(timespent, axis=0))
        total_time_in_quadrant = np.sum(np.divide(timespent,frames_per_second),0)                  # Total time spent in each quadrant
        total_number_of_entries = np.sum(number_of_quadrant_crossing > 0, 0)                       # Total number of entries in each quadrant

        # Alternative method using a function found here:
        # https://stackoverflow.com/questions/41577705/how-does-2d-kernel-density-estimation-in-python-sklearn-work
        # xx, yy, zz = kde2D(x, y, 1.0, 50j, 100j)
        # fig1 = plt.figure()
        # plt.pcolormesh(xx, yy, zz)
        # plt.scatter(x, y, s=0.01, facecolor='white')
        # plt.show()
        
        plt.hist(displacement_raw, 400, density=True, facecolor='g', alpha=0.75)
        
        analyse_results = {'video_width'         : video_width,
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
                           'moviments'           : moviments,
                           'time_moviments'      : time_moviments,
                           'time_resting'        : time_resting
                           }
                           
        return analyse_results
        
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
            try:
                raw_data = pd.read_csv(files.directory[index] + '.csv', sep = ',', na_values = ['no info', '.'], header = None)
                experiments[index].data = raw_data.interpolate(method='spline', order=1, limit_direction = 'both', axis = 0)
            except:
                print("Não existe arquivo CSV com o nome " + files.name[index])
            try:
                experiments[index].last_frame = rgb2gray(skimage.io.imread(files.directory[index] + '.png'))
            except:
                print("Não existe arquivo PNG com o nome " + files.name[index])
                
        return experiments
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    