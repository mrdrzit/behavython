import tkinter as tk
from tkinter import filedialog
import pandas as pd
import imageio as iio
import os

class experiment_class:
    def __init__(self):
        self.name = []
        self.data_directory = []
        self.image_directory = []
        
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
        experiments[index].last_frame = iio.imread(files.directory[index] + '.png')
    except:
        print("Não existe arquivo PNG com o nome " + files.name[index])


        
        
        
        
        
        
        
        
        

