import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go



def dataset_plot(data):

    # Create a single figure with two subplots
    plt.figure(figsize=(12, 6))

    #plt.subplots(1, 2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.heatmap(data, yticklabels=False, xticklabels=False, cbar=True)
    plt.title('Gene expression plot')
    plt.xlabel('Genes')
    plt.ylabel('Cells')


    # Create the KDE plot in the second subplot
    plt.subplot(1, 2, 2)  # Create a new subplot for the KDE plot
    sns.kdeplot(data.sum(axis=0))
    plt.title('Density of total expression throughout the dataset for each Gene')
    plt.xlabel('Sum of Expression')
    plt.ylabel('Density')



    plt.tight_layout()  # Ensure plots don't overlap
    plt.show()