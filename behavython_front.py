import sys
import behavython_back
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow
import pandas as pd

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
        self.analyze_button.clicked.connect(self.analyze_function)
        
    def analyze_function(self):
        behavython_back.plot_viewer_function()
        self.resume_lineedit.clear()
        self.options['arena_width'] = int(self.arena_width_lineedit.text())
        self.options['arena_height'] = int(self.arena_height_lineedit.text())
        self.options['frames_per_second'] = float(self.frames_per_second_lineedit.text())
        if self.animal_combobox.currentIndex() == 0:
            self.options['threshold'] = 0.0267
        else:
            self.options['threshold'] = 0.0267
        
        functions = behavython_back.interface_functions()
        self.experiments = functions.get_experiments(self.resume_lineedit)
        
        for i in range(0, len(self.experiments)):
            self.analyse_results, data_frame = self.experiments[i].video_analyse(self.options)
            if i == 0:
                self.results_data_frame = data_frame
            else:
                self.results_data_frame = self.results_data_frame.join(data_frame, how='outer')
            self.experiments[i].plot_analyse(self.plot_viewer, i)
            self.progress_bar.setValue(round(((i+1)/len(self.experiments))*100))
        
        self.results_data_frame.to_excel(self.experiments[0].directory + '_rusults.xlsx')
                
    def clear_function(self):
        self.type_combobox.setCurrentIndex(1)
        self.frames_per_second_lineedit.setText('30')
        self.arena_width_lineedit.setText('65')
        self.arena_height_lineedit.setText('65')
        self.animal_combobox.setCurrentIndex(0)

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
        
        

