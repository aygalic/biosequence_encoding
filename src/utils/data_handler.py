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

from utils import feature_selection

# datasets

# default path of the folder containing the salmon files
absolute_path = '/Users/aygalic/OneDrive/polimi/Thesis/data/quant/'  
metadata_path = '/Users/aygalic/OneDrive/polimi/Thesis/METADATA_200123.xlsx'  





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
                     metadata_path = metadata_path,
                     feature_selection_threshold = None, 
                     batch_size = 64, 
                     subsample = None, 
                     return_filenames = False,
                     retain_phases = "Both",
                     feature_selection_proceedure = None,
                     sgdc_params = None):

    # getting entries ready
    # each couple of entries correspond to one patient, we are only interested in the "transcript" files
    entries = os.listdir(path)
    entries_transcripts = [e for e in entries if "transcripts" in e ]
    
    # we load metadata, so we can have access to additional information not included in the filename
    meta_data = pd.read_excel(metadata_path, header = 1, usecols = range(1,10) )

    ###########################################
    ###### pre-loading patient selection ######
    ###########################################
    # selecting which entires to include in our analysis

    # To avoid the natural tendency of the model to base its response to different phases
    # we provide the option to focus our analysis on either or both phases of the study.
    if(retain_phases == "1"):
        entries_transcripts = [e for e in entries_transcripts if "Phase1" in e ]
        print("retained phase 1")
    elif(retain_phases == "2"):
        entries_transcripts = [e for e in entries_transcripts if "Phase2" in e ]
        print("retained phase 2")
    elif(retain_phases == "Both"):
        print("retained phases 1 & 2")
    else:
        print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts


    # if we want a smaller dataset for testing purposes
    if(subsample is not None):
        entries_transcripts = entries_transcripts[0:subsample]

    # sanity check : are the patient numbers actually numeric ? 
    entries_transcripts = [e for e in entries_transcripts if e.split(".")[1].isnumeric() ]

    # sanity check : don't load patient where some values are missing
    Na_s =  meta_data[meta_data.isna().any(axis=1)]["Patient Number"]
    entries_transcripts = [e for e in entries_transcripts if e.split(".")[1] not in str(Na_s) ]

    ###########################################
    ############ loading patients  ############
    ###########################################

    # load the dataset into an array 
    print("loading samples...")
    data = [load_patient_data(e) for e in entries_transcripts]

    # remove artifacts by keeping samples of correct length
    samples_to_keep = [1 if s.shape == (95309) else 0 for s in data]
    print("loaded",len(samples_to_keep), "samples")
    
    train_ds = [sample for (sample, test) in  zip(data, samples_to_keep) if test]

    patient_id = [int(p.split(".")[1]) for (p, test) in  zip(entries_transcripts, samples_to_keep) if test]

    # only keep metadata for selected patients
    meta_data = meta_data.set_index('Patient Number')
    meta_data = meta_data.reindex(index=patient_id)
    meta_data = meta_data.reset_index()


    # for each patient in our dataset, we want to know to what cohort he belongs
    cohorts = np.array(meta_data["Cohort"])
    cohorts = [int(value) for value in cohorts]


    ###########################################
    ############ feature selection  ###########
    ###########################################

    # if feature selection is applied
    if(feature_selection_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",feature_selection_threshold, "...")
        data_array = np.array(train_ds)
        gene_selected = feature_selection.MAD_selection(data_array, feature_selection_threshold)
        train_ds = data_array[:,gene_selected]

    if(feature_selection_proceedure == "LASSO"):
        print("selecting genes based on LASSO-like classification...")
        data_array = np.array(train_ds)
        gene_selected = feature_selection.LASSO_selection(data_array, cohorts, sgdc_params)
        train_ds = data_array[:,gene_selected]

    print("number of genes selected : ", len(train_ds[0]))
    x_train = tf.data.Dataset.from_tensor_slices(train_ds)
    dataset = x_train.batch(batch_size)
    if(return_filenames):
        filenames = [f for (f, test) in  zip(entries_transcripts, samples_to_keep) if test]
        return dataset, filenames, len(train_ds[0])
    return dataset, len(train_ds[0])
