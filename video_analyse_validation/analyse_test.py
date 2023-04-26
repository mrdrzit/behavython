import numpy as np
import pandas as pd
import os

# import animal_2_MATLAB.xlsx and animal_2_PYTHON.xlsx then compare the second column of each file against each other
# if they are the same, then print out "TEST PASSED" else if they are not the same, print out "TEST FAILED"

path_python = os.path.dirname(__file__) + "\\animal_2_PYTHON.xlsx"
path_matlab = os.path.dirname(__file__) + "\\animal_2_MATLAB.xlsx"
matlab = pd.read_excel(path_python, header=None)
python = pd.read_excel(path_matlab, header=None)


def test_analyse():
    for i in range(0, len(matlab[1])):
        assert round(matlab[1][i], 10) == round(python[1][i], 10), "TEST FAILED for " + matlab[0][i]
