#!~/anaconda3/bin/python
# Imports
#%%
import general as general
import importlib
importlib.reload(general)
dir(general)

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Read File
df = general.folder("/Dataset/pca_demo.txt")

# Extracting the label from dataset
class_label = general.extractLabel(df)
df = general.extractDataset(df)

# Normalizing the data
df = df.sub(df.mean(axis=0), axis=1)

# Calculating Covariance Matrix
df_mat = np.asmatrix(df)
s = np.cov(df_mat.T)

# Eigen Values & Eigen Vectors
eigenVal, eigenVec = np.linalg.eig(s)

# Sorting of Eigen Vec & Vals
sorted_index = eigenVal.argsort()[::-1] 
eigenVal = eigenVal[sorted_index]
eigenVec = eigenVec[:, sorted_index]

# Forming a reduced dimensional vectors
eigVec = eigenVec[:, :2]
transformed = df_mat.dot(eigVec)

# Attaching the labels to the vectors
final_dataframe = general.horizontalStack(transformed, class_label)

# Convert the numpy array to data frame
final_dataframe = pd.DataFrame(final_dataframe)

# Defining the column names
final_dataframe.columns = ['x','y','label']

# Changing Complex values to real values
A = [x.real for x in final_dataframe.x]
B = [x.real for x in final_dataframe.y]

# Adding Values to dictionary 
dictFrame = {'X': A, 'Y': B}

# Converting dictionary to a dataframe
dFrame = pd.DataFrame(dictFrame)

# Appending label column to the dataframe
changed_dataFrame = general.horizontalStack(dFrame, class_label)

# Changing the array to dataframe and changing the column labels
changed_dataFrame = pd.DataFrame(changed_dataFrame)
changed_dataFrame.columns = ['x', 'y', 'label']

# Plotting the clusters
general.plot2D(changed_dataFrame)

#%%

