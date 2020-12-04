#%%
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import general as general
import importlib
importlib.reload(general)
dir(general)

# Read File
df = general.folder("/Dataset/pca_b.txt")

# Extracting the label from dataset
class_label = general.extractLabel(df)
df = general.extractDataset(df)

sne = TSNE(n_components=3).fit_transform(df)

df_transformed = pd.DataFrame(sne)
final_dataframe = general.horizontalStack(df_transformed, class_label)

final_dataframe = pd.DataFrame(final_dataframe)

# Defining the column names
final_dataframe.columns = ['x', 'y', 'z', 'label']

# Changing Complex values to real values
A = [x.real for x in final_dataframe.x]
B = [x.real for x in final_dataframe.y]
C = [x.real for x in final_dataframe.z]

# Adding Values to dictionary 
dictFrame = {'X': A, 'Y': B, 'Z':C }

# Converting dictionary to a dataframe
dFrame = pd.DataFrame(dictFrame)

# Appending label column to the dataframe
changed_dataFrame = general.horizontalStack(dFrame, class_label)

# Changing the array to dataframe and changing the column labels
changed_dataFrame = pd.DataFrame(changed_dataFrame)
changed_dataFrame.columns = ['x', 'y', 'z', 'label']

general.plot3D(changed_dataFrame)

#%%
