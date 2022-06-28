import sys
import os
import behavython_back
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

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
            self.experiments[i].plot_analysis(self.plot_viewer, i)
            self.progress_bar.emit(round(((i+1)/len(self.experiments))*100))
        
        results_data_frame.to_excel(self.experiments[0].directory + '_rusults.xlsx')
        True_path = os.path.dirname(__file__) + '\\Video_analyse_validation\\animal_2_PYTHON.xlsx'
        results_data_frame.to_excel(True_path, header=False)
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

        super(behavython_gui, self).__init__()              # Calls the inherited classes __init__ method
        uic.loadUi("Behavython_GUI.ui", self)               # Loads the interface design archive (made in Qt Designer)
        self.show()

        self.options = {}
        
        self.clear_button.clicked.connect(self.clear_function)        
        self.analysis_button.clicked.connect(self.analysis_function)
        
    def analysis_function(self):
        plot_viewer_function()
        self.resume_lineedit.clear()
        
        if self.type_combobox.currentIndex() == 0:
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


# def main(): 
#   app = QtWidgets.QApplication(sys.argv)   # Create an instance of QtWidgets.QApplication
#   window = behavython_gui()                # Create an instance of our class
#   app.exec_()                              # Start the application
 
# if __name__ == '__main__': 
#   show = main()


if __name__ == '__main__':   
    def main():
        if not QtWidgets.QApplication.instance():
            QtWidgets.QApplication(sys.argv)
        else:
            QtWidgets.QApplication.instance()
            main = behavython_gui()
            main.show()
            return main  
    show = main()