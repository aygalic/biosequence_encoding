import sys
import matplotlib.pyplot as plt
import numpy as np

# Silhouette score of original groups
from sklearn.metrics import silhouette_score
from utils.benchmark import dunn_index
from utils.benchmark import davies_bouldin

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


sys.path.append('../src')

from utils import benchmark
from utils import visualisation



def auto_kMean(compressed_dataframe):
    # Define a range of cluster numbers to consider
    k_range = range(2, 25) 

    # Create an empty list to store the within-cluster sum of squares (WCSS) for each K
    wcss = []
    silhouette_scores = []

    # Iterate over each K and compute WCSS
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(compressed_dataframe)
        wcss.append(kmeans.inertia_)
        cluster_labels = kmeans.fit_predict(compressed_dataframe)

        silhouette_avg = silhouette_score(compressed_dataframe, cluster_labels)
        silhouette_scores.append(silhouette_avg)

      
    # Plot the WCSS for different values of K
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show()


    # Find the "elbow" point by looking for a change in the rate of decrease in WCSS

    # Choose the optimal number of clusters based on the "elbow" point
    optimal_k = k_range[np.argmax(silhouette_scores)]


    # Choose the optimal number of clusters based on the elbow point
    #optimal_k = 3  # You can manually choose the point that looks like the "elbow" in the plot

    # Perform K-means clustering with the optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    kmeans.fit(compressed_dataframe)

    # Get the cluster labels for each data point
    cluster_labels_kmeans = kmeans.labels_


    TSNE_params = {
                "early_exaggeration" : 5,
                "learning_rate" : 500, 
                "perplexity" : 20, 
                "min_grad_norm" : 1e-7, 
                "n_iter" : 1000,
                "n_components" : 2
            }
    return cluster_labels_kmeans



def auto_DBSCAN(compressed_dataframe):
    dbscan = DBSCAN(eps = 0.00001)
    dbscan.fit(compressed_dataframe)

    # Get the cluster labels (-1 represents noise/outliers)
    cluster_labels = dbscan.labels_

    # Number of clusters found (excluding noise)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Number of clusters: {n_clusters}")
    return cluster_labels
