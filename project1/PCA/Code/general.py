import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def folder(pathname):
    path = os.getcwd()
    file = path + pathname
    df = pd.read_csv(file, delimiter="\t", header=None)
    return df

def extractLabel(df):
    # Extracting the label from dataset
    class_label = pd.DataFrame(df.iloc[:, -1])
    class_label.columns = ['label']
    return class_label

def extractDataset(df):
    df = df.iloc[:, :-1]
    return df

def horizontalStack(transformed, class_label):
    final_dataframe = np.hstack((transformed, class_label))
    return final_dataframe

def plot2D(final_dataframe):
    groups = final_dataframe.groupby('label')
    figure, axes = plt.subplots()
    axes.margins(0.05)
    for disease, data in groups:
        axes.plot(data.x, data.y, marker='o', linestyle='', ms=6, label=disease)
        axes.set_title("PCA.txt")
    axes.legend()
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()

def plot3D(final_dataframe):
    fig = px.scatter_3d(final_dataframe, x='x', y='y', z='z', color='label')
    fig.show()
