import behavython.behavython_back as behavython_back
import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QObject, QThread, pyqtSignal

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
        '''       
        Issue: when the program try to run the ui file, it throws an error. Because modules are not founded in sys.path.
        
        
        Accordingly Python documentation: When a module named spam is imported, the interpreter first searches for a built-in 
        module with that name. These module names are listed in sys.builtin_module_names. If not found, it then searches for 
        a file named spam.py in a list of directories given by the variable sys.path. sys.path is initialized from these locations:
        - The directory containing the input script (or the current directory when no file is specified).
        - PYTHONPATH (a list of directory names, with the same syntax as the shell variable PATH).
        - The installation-dependent default (by convention including a site-packages directory, handled by the site module).

        Solution: To arround this error, I added the directory that contains the behavythn_back and behavython_plot_widget 
        modules.
        '''
        #sys.path.append(os.path.dirname(__file__))
        uic.loadUi(load_gui_path, self)                                             # Loads the interface design archive (made in Qt Designer)
        self.show()

        self.options = {}
        
        self.clear_button.clicked.connect(self.clear_function)        
        self.analysis_button.clicked.connect(self.analysis_function)
        
    def analysis_function(self):
        self.clear_plot()
        self.resume_lineedit.clear()
        
        if self.type_combobox.currentIndex() == 1:
            self.options['experiment_type'] = 'open_field'
        else:
            self.options['experiment_type'] = 'plus_maze'
        self.options['arena_width'] = int(self.arena_width_lineedit.text())
        self.options['arena_height'] = int(self.arena_height_lineedit.text())
        self.options['frames_per_second'] = float(self.frames_per_second_lineedit.text())
        if self.animal_combobox.currentIndex() == 0:
            self.options['threshold'] = 0.0267  # Motion detection threshold (mice)
        else:
            self.options['threshold'] = 0.0667  # Motion detection threshold (rats)
        
        functions = behavython_back.interface_functions()
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
        self.clear_plot()

    def clear_plot(self):
        for i in range(1,10):
            self.plot_viewer.canvas.axes[i-1].cla()                                             # Changes the plot face color


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