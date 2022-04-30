import os
import skimage.io
import tkinter as tk
import pandas as pd
import numpy as np
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
        velocity = np.divide(displacement, np.transpose(np.append(0, np.diff(time_vector)))) # Expand steps to make the code more readable
        mean_velocity = np.nanmean(velocity)
        
        aceleration = np.divide(np.append(0, np.diff(velocity)), np.append(0, np.diff(time_vector)))
        moviments = np.sum(displacement > 0)
        time_moviments = np.sum(displacement > 0)*(1/frames_per_second)
        time_resting = np.sum(displacement == 0)*(1/frames_per_second)

        xy = np.array([np.array(x_axe), np.array(y_axe)])
        kde = stats.gaussian_kde(xy)
        pdf = np.array(kde.evaluate(xy)) # # Convert to to double and compare
        color_limits = np.array([(x - np.min(pdf))/(np.max(pdf) - np.min(pdf)) for x in pdf])
        fig = plt.scatter(y_axe, x_axe, c=color_limits, cmap = 'jet', linewidths=0.1, marker='.')
        plt.show()
        pass

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    