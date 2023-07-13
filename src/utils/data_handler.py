import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy


from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras


# datasets

# default path of the folder containing the salmon files
absolute_path = '/Users/aygalic/OneDrive/polimi/Thesis/data/quant/'  





# from filename to tensor
# here we open a single file passed as "filename" we return a tensor of TPM values.
def load_patient_data(filename):
  #specify read types for our data
  read_types = [float()]
  # get a first sample to base everything of
  text = pathlib.Path(absolute_path + filename).read_text()
  lines = text.split('\n')[1:-1]
  # the 3rd column correspond to TPM values.
  features = tf.io.decode_csv(lines, record_defaults=read_types, field_delim = "\t", select_cols=[3])
  data = tf.convert_to_tensor(features)[0]
  return data


# building the actual dataset
def generate_dataset(path = absolute_path, 
                     feature_selection_threshold = None, 
                     batch_size = 64, 
                     subsample = None, 
                     return_filenames = False,
                     retain_phases = "Both"):
    # getting entries ready
    # each couple of entries correspond to one patient, we are only interested in the "transcript" files
    entries = os.listdir(absolute_path)
    entries_transcripts = [e for e in entries if "transcripts" in e ]

    # retain only the phase(s) of interest
    if(retain_phases == "1"):
        entries_transcripts = [e for e in entries if "Phase1" in e ]
        print("retained phase 1")
    elif(retain_phases == "2"):
        entries_transcripts = [e for e in entries if "Phase2" in e ]
        print("retained phase 2")
    elif(retain_phases == "Both"):
        print("retained phases 1 & 2")
    else:
        print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts


    # if we want a smaller dataset
    if(subsample is not None):
        entries_transcripts = entries_transcripts[1:subsample]

    # load the dataset into an array 
    print("loading samples...")
    data = [load_patient_data(e) for e in entries_transcripts]

    # remove artifacts by keeping samples of correct length
    samples_to_keep = [1 if s.shape == (95309) else 0 for s in data]
    print("loaded",len(samples_to_keep), "samples")
    train_ds = [sample for (sample, test) in  zip(data, samples_to_keep) if test]

    # if feature selection is applied
    if(feature_selection_threshold is not None):
        data_array = np.array(train_ds)
        MAD = scipy.stats.median_abs_deviation(data_array)
        gene_selected = [True if val > feature_selection_threshold else False for val in MAD]
        print("number of genes selected : ",sum(gene_selected))
        train_ds = data_array[:,gene_selected]

    x_train = tf.data.Dataset.from_tensor_slices(train_ds)
    dataset = x_train.batch(batch_size)
    if(return_filenames):
        filenames = [f for (f, test) in  zip(entries_transcripts, samples_to_keep) if test]
        return dataset, filenames
    return dataset
