import behavython_back
import sys
import os
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow, QMessageBox
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
                self.experiments[i].plot_analysis_open_field(self.plot_viewer, i, self.options['save_folder'])
            else:
                self.experiments[i].plot_analysis_pluz_maze(self.plot_viewer, i, self.options['save_folder'])
            self.progress_bar.emit(round(((i+1)/len(self.experiments))*100))
        
        if self.options['plot_options'] == 1:
          results_data_frame.to_excel(self.options['save_folder'] + '/analysis_results.xlsx')
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
        load_gui_path = os.path.dirname(__file__) + "\\behavython_gui.ui"
        uic.loadUi(load_gui_path, self)                                             # Loads the interface design archive (made in Qt Designer)
        self.show()

        self.options = {}
        
        self.clear_button.clicked.connect(self.clear_function)        
        self.analysis_button.clicked.connect(self.analysis_function)
        
    def analysis_function(self):
        self.resume_lineedit.clear()
        self.options['arena_width'] = int(self.arena_width_lineedit.text())
        self.options['arena_height'] = int(self.arena_height_lineedit.text())
        self.options['frames_per_second'] = float(self.frames_per_second_lineedit.text())
        self.options['experiment_type'] = self.type_combobox.currentText().lower().strip().replace(' ', '_')             # Set the experiment type. Convert to lowercase, remove spaces and replace with underscores to match the naming convention
        self.options['plot_options'] = self.save_button.isChecked()                                                      
        self.options['max_fig_res' ] = str(self.fig_max_size.currentText()).replace(' ','').replace('x',',').split(',')  # Remove trailing spaces and replace x with comma and split the values at the comma to make a list
        if self.animal_combobox.currentIndex() == 0:
            self.options['threshold'] = 0.0267  # Motion detection threshold (mice)
        else:
            self.options['threshold'] = 0.0667  # Motion detection threshold (rats)
        
        functions = behavython_back.interface_functions()
        [self.experiments, save_folder, error_flag, inexistent_file] = functions.get_experiments(self.resume_lineedit, self.options['experiment_type'], self.options['plot_options'])
        if error_flag != 0:
          if error_flag == 1:  
            warning_message_function("File selection problem", "WARNING!! No files were selected. Please select a file and try again.")
          elif error_flag == 2:
            warning_message_function("File selection problem", "WARNING!! No destination folder was selected. Please select a valid folder and try again.") # Sets the text of the message box
          elif error_flag == 3:
            warning_message_function("File selection problem", "WARNING!! Doesn't exist a CSV or PNG file with the name " + inexistent_file)
          sys.exit(4)
        self.options['save_folder'] = save_folder                                                 
            
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
        self.fig_max_size.setCurrentIndex(0)
        self.clear_plot()

    def clear_plot(self):
        for i in range(1,10):
            self.plot_viewer.canvas.axes[i-1].cla()                                             # Changes the plot face color
    
def warning_message_function(title, text):
    warning = QMessageBox()                                                                                 # Create the message box
    warning.setWindowTitle(title)                                                                           # Message box title
    warning.setText(text)                                                                                   # Message box text
    warning.setIcon(QMessageBox.Warning)                                                                    # Message box icon
    warning.setStyleSheet("QMessageBox{background:#353535;}QLabel{font:10pt/DejaVu Sans/;" +
                "font-weight:bold;color:#FFFFFF;}QPushButton{width:52px; border:2px solid #A21F27;border-radius:8px;" +
                "background-color:#2C53A1;color:#FFFFFF;font:10pt/DejaVu Sans/;" +
                "font-weight:bold;}QPushButton:pressed{border:2px solid #A21F27;" +
                "border-radius:8px;background-color:#A21F27;color:#FFFFFF;}")
    warning.setStandardButtons(QMessageBox.Ok)                                                              # Message box buttons
    warning.exec_()


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