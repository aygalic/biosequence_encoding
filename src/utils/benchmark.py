# sup
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import losses



from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors



def count_parameters(model):
    ''' for torch models, print number of param. '''
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    return total_params, trainable_params


def hopkins(data, n=None):
    if n is None:
        n = len(data)
    
    # Step 1: Randomly sample n data points from data
    sample = data[np.random.choice(data.shape[0], n, replace=False)]
    
    # Step 2: Nearest neighbor distances for real data
    nbrs = NearestNeighbors(n_neighbors=2).fit(data)
    _, indices = nbrs.kneighbors(sample)
    w_distances = np.array([np.linalg.norm(sample[i] - data[indices[i, 1]]) for i in range(n)])
    
    # Step 3: Generate n random data points
    mins, maxs = np.min(data, axis=0), np.max(data, axis=0)
    random_data = np.array([np.random.uniform(mins[i], maxs[i], n) for i in range(data.shape[1])]).T
    
    # Step 4: Nearest neighbor distances for random data
    _, indices = nbrs.kneighbors(random_data)
    u_distances = np.array([np.linalg.norm(random_data[i] - data[indices[i, 0]]) for i in range(n)])
    
    # Step 5: Calculate Hopkins statistic
    H = np.sum(u_distances) / (np.sum(u_distances) + np.sum(w_distances))
    return H





#### Clustering related

# Dunn Index
def dunn_index(data, labels):
    cluster_centers = []

    for cluster_label in np.unique(labels):
        cluster_data = data[labels == cluster_label]
        cluster_center = np.mean(cluster_data, axis=0)
        cluster_centers.append(cluster_center)
    
    max_diameter = 0.00000000001 # quick fix
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            diameter = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            if diameter > max_diameter:
                max_diameter = diameter

    min_distance = np.inf
    for i in range(len(cluster_centers)):
        for j in range(i + 1, len(cluster_centers)):
            distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
            if distance < min_distance:
                min_distance = distance

    dunn_index = min_distance / max_diameter
    return dunn_index


# Davies-Bouldin Index
def davies_bouldin(data, labels):
    unique_labels = np.unique(labels)
    cluster_centers = []
    cluster_variances = []

    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_center = np.mean(cluster_data, axis=0)
        cluster_centers.append(cluster_center)
        variance = np.mean(np.linalg.norm(cluster_data - cluster_center, axis=1))
        cluster_variances.append(variance)

    db_index = 0
    for i in range(len(unique_labels)):
        max_db = -np.inf
        for j in range(len(unique_labels)):
            if i != j:
                db = (cluster_variances[i] + cluster_variances[j]) / np.linalg.norm(cluster_centers[i] - cluster_centers[j])
                if db > max_db:
                    max_db = db
        db_index += max_db

    db_index /= len(unique_labels)
    return db_index

def print_metrics(data, labels):
    # we also want to plot metrics :

    silhouette_avg = silhouette_score(data, labels)
    print(f"Silhouette Score: {silhouette_avg:.4f}")


    dunn = dunn_index(data, labels)
    print(f"Dunn Index: {dunn:.4f}")


    db = davies_bouldin(data, labels)
    print(f"Davies-Bouldin Index: {db:.4f}")