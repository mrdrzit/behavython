from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class plot_viewer(QtWidgets.QWidget):
    """
    This class modifies the interface's QWidget in order to insert a plot viewer.

    TODO: Find the best way to plot a summary of the data.
    E.g.: A button to advance or go back in the plots
    """

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure(facecolor="#353535", dpi=100, tight_layout=True))  # Create a figure object

        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding
        )  # Creates axes size policy
        self.canvas.setSizePolicy(sizePolicy)  # Sets the size policy

        self.canvas.axes = []
        for i in range(1, 10):
            self.canvas.axes.append(self.canvas.figure.add_subplot(3, 3, i))  # Creates an empty plot
            self.canvas.axes[i - 1].set_facecolor("#252525")  # Changes the plot face color
            self.canvas.axes[i - 1].get_xaxis().set_visible(False)
            self.canvas.axes[i - 1].get_yaxis().set_visible(False)

        vertical_layout = QtWidgets.QVBoxLayout()  # Creates a layout
        vertical_layout.addWidget(self.canvas)  # Inserts the figure on the layout
        self.canvas.figure.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)  # Sets the plot margins
        self.setLayout(vertical_layout)  # Sets the layout

        def includeFile(self):
            return "QtCustomWidgets.widgets.mybutton"
