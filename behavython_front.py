import sys
import behavython_back
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QMainWindow

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
        self.options['arena_width'] = int(self.arena_width_lineedit.text())
        self.options['arena_height'] = int(self.arena_height_lineedit.text())
        self.options['frames_per_second'] = float(self.frames_per_second_lineedit.text())
        self.options['threshold'] = float(self.threshold_lineedit.text())
        
        functions = behavython_back.interface_functions()
        self.experiments = functions.get_experiments()
        
        self.analyse_results = self.experiments[0].video_analyse(self.options)
        print(self.analyse_results)
        self.experiments[0].plot_analyse(self.plot_viewer)
          
    def clear_function(self):
        self.type_combobox.setCurrentIndex(1)
        self.frames_per_second_lineedit.setText('30')
        self.arena_width_lineedit.setText('65')
        self.arena_height_lineedit.setText('65')
        self.threshold_lineedit.setText('0.0267')

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
        
        

