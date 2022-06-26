import numpy as np
import pandas as pd

# import animal_2_MATLAB.xlsx and animal_2_PYTHON.xlsx then compare the second column of each file against each other
# if they are the same, then print out "TEST PASSED" else if they are not the same, print out "TEST FAILED" 


matlab = pd.read_excel("Behavython/Video_analyse_validation/animal_2_MATLAB.xlsx", header=None)
python = pd.read_excel("Behavython/Video_analyse_validation/animal_2_PYTHON.xlsx", header=None)

def test_analyse():
  for i in range(0, len(matlab[1])): 
    assert round(matlab[1][i], 10) == round(python[1][i], 10), "TEST FAILED for " + matlab[0][i]