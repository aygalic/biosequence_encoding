# sup
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import losses



def brenchmark(model, dataset, param_set):
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

    ############
    ###### COMPUTE METRICS
    ############
    # compute the loss of the model over the whole dataset
    # dataset has to be normalized in the range [0,1]
    z = model.encoder(data)
    reconstruction = model.decoder(z)
    total_loss = tf.reduce_mean(losses.mean_squared_error(data, reconstruction), axis=(0)).numpy()
    print("loss:", total_loss)

    ############
    ###### PERFORM COMPARAISON
    ############
    # put everything into a dataframe
    curr_bench = pd.DataFrame({"model_name" : name,
                               "param_count" : n,
                               "loss" : total_loss}, index=[0])

    csv_file_path = '../workfiles/benchmark_history.csv'

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

    return df

