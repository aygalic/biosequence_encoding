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



def count_parameters(model):
    ''' for torch models, print number of param. '''
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

def benchmark(model, dataset, dataset_metadata):

    ############
    ###### GATHER INFO
    ############
    # basic informations about model 
    name = dataset_metadata["name"]
    print("benchmarking model :", name)

    # get the total number of parametter
    n = model.encoder.count_params() + model.encoder.count_params()
    print("number of parametters of the model:", n)

    # compute the reconstruction err on the whole dataset
    data = dataset
    
    # we want some basic info about the data:
    print("shape of the dataset:", data.shape)



    obs_count = dataset_metadata["n_seq"]
    gene_count = dataset_metadata["n_features"]


    # check if the dataset has the correct range : 
    if(np.amin(data) < 0):
        print("the dataset is NOT meeting requirement : min(data) =", min(data))
        return None
    if(np.amax(data) > 1):
        print("the dataset is NOT meeting requirement : max(data) =", max(data))
        return None

    print("the dataset meets the min_max requirement")
    


    ############
    ###### COMPUTE METRICS
    ############
    # compute the loss of the model over the whole dataset
    # dataset has to be normalized in the range [0,1]
    if(model._is_variational == True):
        _,__, z = model.encoder(data)
    else :
        z = model.encoder(data)

    reconstruction = model.decoder(z)
    reconstruction = model.decoder(z) # in case of variational autoencoder
    
    if(dataset._is_time_series):
        total_loss = tf.reduce_mean(tf.reduce_mean(losses.mean_squared_error(data, reconstruction), axis=(0))).numpy()
    else:
        total_loss = tf.reduce_mean(losses.mean_squared_error(data, reconstruction), axis=(0)).numpy()

    print("loss:", total_loss)

    ############
    ###### PERFORM COMPARAISON
    ############
    # put everything into a dataframe
    curr_bench = pd.DataFrame({"model_name" : name,
                               "param_count" : n,
                               "loss" : total_loss,
                               "obs_count" : obs_count,
                               "gene_number" : gene_count}, index=[0],)


    csv_file_path = '../workfiles/benchmark_history'+dataset._name+'.csv'

    # Check if the file exists
    try:
        df = pd.read_csv(csv_file_path)
        print("adding new performer to the history")
        df = pd.concat([df, curr_bench], axis = 0)

    except FileNotFoundError:
        df = curr_bench
        print("this is the first entry to the benchmark history")
        
    # Save the DataFrame to the  CSV file
    df.to_csv(csv_file_path, index=False)

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=df,
        x="param_count", y="loss", size="param_count",
        sizes=(10, 200),
    )
    #g.set(xscale="log", yscale="log")
    g.ax.xaxis.grid(True, "minor", linewidth=.25)
    g.ax.yaxis.grid(True, "minor", linewidth=.25)
    g.despine(left=True, bottom=True)
    g

    return df






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