"""Importing libraries"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skimage import measure

"""Get current working directory"""
wd = os.getcwd() #need to set wd as the file right above the level of the individual questions


"""Loading datasets + checking basic characteristics of dataset"""
q4_wd = os.path.join(wd, "Question 4")

#Reading files as Pandas df
image_1 = pd.read_csv(os.path.join(q4_wd, "input_question_4"), sep='\t', names = range(0,20))

#checking characteristics
print("Shape of input image:", image_1.shape)
image_1


"""Identifying the 4-connectivity pattern of input image"""
#measure.label() is the function which takes in matrix and returns connectivity patterns of the input matrix.
all_labels = measure.label(image_1, connectivity = 1)

print(all_labels)

"""Creation of output file for submission"""
np.savetxt(r"Question 4/output_question_4.txt", all_labels, fmt = '%.d')