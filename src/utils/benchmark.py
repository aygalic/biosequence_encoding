# sup
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import losses



def benchmark(model, dataset):

    ############
    ###### GATHER INFO
    ############
    # basic informations about model 
    name = model._name
    print("benchmarking model :", name)

    # get the total number of parametter
    n = model.encoder.count_params() + model.encoder.count_params()
    print("number of parametters of the model:", n)

    # compute the reconstruction err on the whole dataset
    data = np.concatenate(list(dataset.as_numpy_iterator()), axis=0)
    
    # we want some basic info about the data:
    print("shape of the dataset:", data.shape)


    if(~dataset._is_transpose and ~dataset._is_time_series):
        obs_count = data.shape[0]
        gene_count = data.shape[1]
    elif(~dataset._is_transpose and dataset._is_time_series):
        obs_count = data.shape[0]
        gene_count = data.shape[2]
        n_time_stamps = data.shape[1]
    elif(dataset._is_transpose and dataset._is_time_series):
        obs_count = data.shape[0]
        gene_count = data.shape[1]
        n_time_stamps = data.shape[2]

    else:
        print("wrong data format")
        return None

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

