import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy

from sklearn.preprocessing import normalize


from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow import keras

from utils import feature_selection

# datasets

# default path of the folder containing the salmon files
absolute_path = '/Users/aygalic/Thesis/data/quant/'  
metadata_path = '/Users/aygalic/Thesis/METADATA_200123.xlsx'  





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
                     retain_phases = None,
                     minimum_time_point = "BL",
                     feature_selection_proceedure = None,
                     sgdc_params = None,
                     class_balancing = None):

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
        print("Retaining patients that are included in phases 1 & 2")
        phase_1_ids = [p.split(".")[1] for p in  entries_transcripts if "Phase1" in p]
        phase_2_ids = [p.split(".")[1] for p in  entries_transcripts if "Phase2" in p]
        # Find the entries that match with both Phase 1 and Phase 2
        common_ids = set(phase_1_ids) & set(phase_2_ids) # set intersection
        entries_matching_both_phases = [entry for entry in entries_transcripts if any(f".{common_id}." in entry for common_id in common_ids)]
        entries_transcripts = entries_matching_both_phases
    else:
        print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts


    # We can decide to only include patient who completed a given quantities of timepoints
    # The following stategy for filtering also filters out every patient who have missed a visit up to the given timepoint.
    # This comportement could be tweaked easely later on
    # a bit clunky though
    if(minimum_time_point == "BL"):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "BL"] 
        matchin_entries = [entry for entry in entries_transcripts if entry.split(".")[1] in BL_ids]
        entries_transcripts = matchin_entries
    elif(minimum_time_point == "V02"):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V02"] 
        common_ids = set(BL_ids) & set(V02_ids) 
        matchin_entries = [entry for entry in entries_transcripts if entry.split(".")[1] in common_ids]
        entries_transcripts = matchin_entries
    elif(minimum_time_point == "V04"):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V04"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) 
        matchin_entries = [entry for entry in entries_transcripts if entry.split(".")[1] in common_ids]
        entries_transcripts = matchin_entries
    elif(minimum_time_point == "V06"):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V04"] 
        V06_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V06"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) 
        matchin_entries = [entry for entry in entries_transcripts if entry.split(".")[1] in common_ids]
        entries_transcripts = matchin_entries
    elif(minimum_time_point == "V08"):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V04"] 
        V06_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V06"] 
        V08_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V08"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) & set(V08_ids) 
        matchin_entries = [entry for entry in entries_transcripts if entry.split(".")[1] in common_ids]
        entries_transcripts = matchin_entries




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
    cohorts = np.array(meta_data["Cohort"], dtype=np.int32)


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
        gene_selected = feature_selection.LASSO_selection(data_array, cohorts, sgdc_params, class_balancing)
        train_ds = data_array[:,gene_selected]

    print("number of genes selected : ", len(train_ds[0]))
    x_train = tf.data.Dataset.from_tensor_slices(train_ds)
    dataset = x_train.batch(batch_size)
    if(return_filenames):
        filenames = [f for (f, test) in  zip(entries_transcripts, samples_to_keep) if test]
        return dataset, filenames, len(train_ds[0])
    return dataset, len(train_ds[0])





### now we design a function that return a dataset of multivriate time series
# building the actual dataset
def generate_timeseries_dataset(path = absolute_path, 
                     metadata_path = metadata_path,
                     feature_selection_threshold = None, 
                     batch_size = 64, 
                     subsample = None, 
                     return_id = False,
                     retain_phases = None,
                     feature_selection_proceedure = None,
                     sgdc_params = None,
                     class_balancing = None):

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
    
    # Actually let's simplify this whole thing for now
    """
    if(retain_phases == "1"):
        entries_transcripts = [e for e in entries_transcripts if "Phase1" in e ]
        print("retained phase 1")
    elif(retain_phases == "2"):
        entries_transcripts = [e for e in entries_transcripts if "Phase2" in e ]
        print("retained phase 2")
    elif(retain_phases == "Both"):
        print("Retaining patients that are included in phases 1 & 2")
        phase_1_ids = [p.split(".")[1] for p in  entries_transcripts if "Phase1" in p]
        phase_2_ids = [p.split(".")[1] for p in  entries_transcripts if "Phase2" in p]
        # Find the entries that match with both Phase 1 and Phase 2
        common_ids = set(phase_1_ids) & set(phase_2_ids) # set intersection
        entries_matching_both_phases = [entry for entry in entries_transcripts if any(f".{common_id}." in entry for common_id in common_ids)]
        entries_transcripts = entries_matching_both_phases
    else:
        print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts
    """



    # we only retain patient who went through all visits
    BL_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "BL"] 
    V02_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V02"] 
    V04_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V04"] 
    V06_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V06"] 
    V08_ids = [p.split(".")[1] for p in  entries_transcripts if p.split(".")[2] == "V08"] 
    common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) & set(V08_ids) 
    matchin_entries = [entry for entry in entries_transcripts if entry.split(".")[1] in common_ids]
    entries_transcripts = matchin_entries








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
    cohorts = np.array(meta_data["Cohort"], dtype=np.int32)


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
        gene_selected = feature_selection.LASSO_selection(data_array, cohorts, sgdc_params, class_balancing)
        train_ds = data_array[:,gene_selected]

    print("number of genes selected : ", len(train_ds[0]))


    ###########################################
    ############## normalisation  #############
    ###########################################

    if(True): # i'll set it as an option when i feel like it'll be reasonable
        print("normalizing data...")
        data_array = np.array(train_ds)
        train_ds = normalize(data_array)
        print("normalization done")

    ##########################################
    ######## Building the time series ########
    ##########################################

    big_bad_dict = {key: [sample for (sample, name) in zip(train_ds, entries_transcripts) if int(name.split(".")[1]) == key] for key in patient_id}

    print("number of seq to be analized :", len(train_ds))
    print("number of actual individual to be studied :", len(big_bad_dict))

    # Convert dictionary values to numpy arrays
    sequences = np.array(list(big_bad_dict.values()), dtype=np.float32)

    # Step 2: Create a TensorFlow Dataset
    x_train = tf.data.Dataset.from_tensor_slices(sequences)

    # to keep track of which timeserie correspond to which identifier
    sequence_names = list(big_bad_dict.keys())

    # same process as previously
    dataset = x_train.batch(batch_size)

    print(dataset.cardinality().numpy())

    if(return_id):
        return dataset, sequence_names, len(train_ds[0])
    return dataset, len(train_ds[0])