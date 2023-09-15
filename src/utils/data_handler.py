import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy

from sklearn.preprocessing import normalize, MinMaxScaler


from tensorflow import keras

from utils import feature_selection

# for translation of gene symbols
import mygene
mg = mygene.MyGeneInfo()

# datasets

# default path of the folder containing the salmon files
absolute_path = '/Users/aygalic/Thesis/data/quant/'  
absolute_path_cancer = '/Users/aygalic/Thesis/data/cancer'  

metadata_path = '/Users/aygalic/Thesis/METADATA_200123.xlsx'  




# from filename to tensor
# here we open a single file passed as "filename" we return a tensor of TPM values.
def load_patient_data(filename, header = 0):
    data = pd.read_table(filename, header = header)
    return data.iloc[:, 3]

# here we open a single file passed as "filename" we return a lit of the values names.
def get_names(filename, header = 0):
    names = pd.read_table(filename, header = header)
    return names.iloc[:, 0]


### now we design a function that return a dataset of multivriate time series or cell wise observations
def generate_dataset(path = absolute_path, 
                     metadata_path = metadata_path,
                     feature_selection_threshold = None, 
                     batch_size = 64, 
                     subsample = None, 
                     retain_phases = None,
                     feature_selection_proceedure = None,
                     sgdc_params = None,
                     class_balancing = None,
                     normalization = True,
                     minimum_time_point = "BL",
                     as_time_series = False,
                     transpose = False,
                     MT_removal = True,
                     log1p = True,
                     min_max = True,
                     keep_only_symbols = False,
                     drop_ambiguous_pos = False,
                     sort_symbols = False,
                     # for experiment purpose only :
                     keep_only_BL = False,
                     keep_only_genetic_pd = False
                     ):

    dataset_of_interest = "genes"

    # getting entries ready
    # each couple of entries correspond to one patient, we are only interested in the "transcript" files
    entries = os.listdir(path)
    #entries_transcripts = [e for e in entries if "transcripts" in e ]
    entries = [e for e in entries if dataset_of_interest in e ]
    
    # we load metadata, so we can have access to additional information not included in the filename
    meta_data = pd.read_excel(metadata_path, header = 1, usecols = range(1,10) )

    ###########################################
    ###### pre-loading patient selection ######
    ###########################################
    # selecting which entires to include in our analysis

    # To avoid the natural tendency of the model to base its response to different phases
    # we provide the option to focus our analysis on either or both phases of the study.
    

    if(retain_phases == "1"):
        entries = [e for e in entries if "Phase1" in e ]
        print("retained phase 1")
    elif(retain_phases == "2"):
        entries = [e for e in entries if "Phase2" in e ]
        print("retained phase 2")
    elif(retain_phases == "Both"):
        print("Retaining patients that are included in phases 1 & 2")
        phase_1_ids = [p.split(".")[1] for p in  entries if "Phase1" in p]
        phase_2_ids = [p.split(".")[1] for p in  entries if "Phase2" in p]
        # Find the entries that match with both Phase 1 and Phase 2
        common_ids = set(phase_1_ids) & set(phase_2_ids) # set intersection
        entries_matching_both_phases = [entry for entry in entries if any(f".{common_id}." in entry for common_id in common_ids)]
        entries = entries_matching_both_phases
    elif(retain_phases == None):
        print("not applying any filtering over phases")
    else:
        print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts


    # if we want a smaller dataset for testing purposes
    if(subsample is not None):
        entries = entries[0:subsample]

    # We can decide to only include patient who completed a given quantities of timepoints
    # The following stategy for filtering also filters out every patient who have missed a visit up to the given timepoint.
    # This comportement could be tweaked easely later on
    # a bit clunky though
    if(minimum_time_point == "BL" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in BL_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V02" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line to month 6 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        common_ids = set(BL_ids) & set(V02_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V04" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line to month 12 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V06" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line to month 24 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        V06_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V06"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    
    # if we want time series, we constrain them to only patients that went through every visits.
    elif(minimum_time_point == "V08" or as_time_series == True):
        print("retaining all patient who have passed all visits...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        V06_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V06"] 
        V08_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V08"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) & set(V08_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries


    #### FOR EXERIMENTATION
    if(keep_only_BL):
        entries = [e for e in entries if e.split(".")[2] == "BL"]




    # sanity check : are the patient numbers actually numeric ? 
    entries = [e for e in entries if e.split(".")[1].isnumeric() ]

    # sanity check : don't load patient where some values are missing
    Na_s =  meta_data[meta_data.isna().any(axis=1)]["Patient Number"]
    entries = [e for e in entries if e.split(".")[1] not in str(Na_s) ]


    #### FOR EXERIMENTATION
    if(keep_only_genetic_pd):
        GPD = [1 if ds == "Genetic PD" else 0 for ds in meta_data["Disease Status"] ]
        entries = entries[GPD]



    ###########################################
    ############ loading patients  ############
    ###########################################

    # load the dataset into an array 
    print("loading samples...")
    data = [load_patient_data(os.path.join(path, e)) for e in entries]

    # get the entry name list
    names = get_names(os.path.join(path,entries[0]))
    
    # getting rid of the version number
    names = [n.split(".")[0] for n in names]
    
    # remove artifacts by keeping samples of correct length

    samples_to_keep = [1 if s.shape == (34569,) else 0 for s in data]
        

    print("loaded",sum(samples_to_keep), "samples")
    
    train_ds = [sample for (sample, test) in  zip(data, samples_to_keep) if test]
    data_array = np.array(train_ds)


    patient_id = [int(p.split(".")[1]) for (p, test) in  zip(entries, samples_to_keep) if test]

    # only keep metadata for selected patients
    meta_data = meta_data.set_index('Patient Number')
    meta_data = meta_data.reindex(index=patient_id)
    meta_data = meta_data.reset_index()


        


    ###########################################
    ############ feature selection  ###########
    ###########################################
    
    print("retriving symbols for genes")
    query_result = mg.querymany(names, fields = ['genomic_pos', 'symbol'], scopes='ensembl.gene', species='human', verbose = False, as_dataframe = True)

    query_result = query_result.reset_index()
    query_result = query_result.drop_duplicates(subset = ["query"])
    # here we have the correct length

    names = [q if(pd.isna(s)) else s for (s,q) in zip(query_result["symbol"],query_result["query"])]
    query_result['name'] = names

    if(MT_removal == True):
        gene_selected = [False if q.startswith('MT') else True for q in query_result['name']]
        print("removing", len(gene_selected) - sum(gene_selected), "mithocondrial genes from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(keep_only_symbols == True):
        gene_selected = [False if s.startswith('ENSG') else True for s in query_result['name']]
        print("removing", len(gene_selected) - sum(gene_selected), "not found symbols from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(drop_ambiguous_pos == True):
        gene_selected = query_result["genomic_pos"].isna()
        print("removing", len(gene_selected) - sum(gene_selected), "ambigously positioned symbols from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(feature_selection_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",feature_selection_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data_array, feature_selection_threshold)
        print("removing", len(gene_selected) - sum(gene_selected), "genes under the MAD threshold from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(feature_selection_proceedure == "LASSO"):
        # for each patient in our dataset, we want to know to what cohort he belongs
        cohorts = np.array(meta_data["Cohort"], dtype=np.int32)
        print("selecting genes based on LASSO-like classification...")
        gene_selected = feature_selection.LASSO_selection(data_array, cohorts, sgdc_params, class_balancing)
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]



    print("number of genes selected : ", len(data_array[0]))


    ###########################################
    ################# sorting  ################
    ###########################################
    if(sort_symbols):
        print("sorting based on genomic position chr then transcript start...")
        # reset the indexes because of all the previous transformations we have done
        query_result = query_result.reset_index(drop=True)
        query_result = query_result.sort_values(['genomic_pos.chr', 'genomic_pos.start'], ascending=[True, True])
        # Extract the sorted rows as a NumPy array
        data_array = data_array[:, query_result.index]


    ###########################################
    ############## normalisation  #############
    ###########################################

    
    if(normalization == True): 
        print("normalizing data...")
        data_array = normalize(data_array)

    if(log1p == True): 
        print("log(1 + x) transformation...")
        data_array = np.log1p(data_array)

    # after log1p transform because it already provide us with a very good dataset 
    if(min_max == True):
        print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data_array = scaler.fit_transform(data_array)


    ##########################################
    ######## Building the time series ########
    ##########################################
    print("number of seq in the dataset :", len(data_array))

    if(as_time_series == True):
        print("converting samples to time series")
        big_bad_dict = {key: [sample for (sample, name) in zip(data_array, entries) if int(name.split(".")[1]) == key] for key in patient_id}

        print("number of actual individual to be studied :", len(big_bad_dict))

        # Convert dictionary values to numpy arrays
        sequences = np.array(list(big_bad_dict.values()), dtype=np.float32)

        # Step 2: Create a TensorFlow Dataset
        x_train = tf.data.Dataset.from_tensor_slices(sequences)

        if(transpose):
            print("using transposed data...")

            # Define a function to transpose a sequence
            def transpose_sequence(sequence):
                return tf.transpose(sequence, perm=[1, 0])

            # Transpose each element in the dataset
            x_train = x_train.map(transpose_sequence)


        # to keep track of which timeserie correspond to which identifier
        sequence_names = list(big_bad_dict.keys())
    else:
        # we don't assemble the files into timeseries and simply return the TPM values and the corresponding filename
        print("keeping sample as is, no conversion to time series")
        x_train = tf.data.Dataset.from_tensor_slices(data_array)
        sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]

    # make it a batched dataset
    dataset = x_train.batch(batch_size)
    
    # adding correct attributes
    dataset._name = "genes"
    dataset._is_transpose = transpose
    dataset._is_time_series = as_time_series

    return dataset, sequence_names, len(data_array[0]), query_result













### now we design a function that return a dataset of multivriate time series or the individual timestamps
def generate_dataset_transcripts(path = absolute_path, 
                     metadata_path = metadata_path,
                     feature_selection_threshold = None, 
                     batch_size = 64, 
                     subsample = None, 
                     retain_phases = None,
                     normalization = True,
                     minimum_time_point = "BL",
                     as_time_series = False,
                     transpose = False,
                     MT_removal = True,
                     log1p = True,
                     min_max = True,
                     # for experiment purpose only :
                     keep_only_BL = False,
                     keep_only_genetic_pd = False):

    dataset_of_interest = "transcripts"



    # getting entries ready
    # each couple of entries correspond to one patient, we are only interested in the "transcript" files
    entries = os.listdir(path)
    #entries_transcripts = [e for e in entries if "transcripts" in e ]
    entries = [e for e in entries if dataset_of_interest in e ]
    # we load metadata, so we can have access to additional information not included in the filename
    meta_data = pd.read_excel(metadata_path, header = 1, usecols = range(1,10) )

    ###########################################
    ###### pre-loading patient selection ######
    ###########################################
    # selecting which entires to include in our analysis

    # To avoid the natural tendency of the model to base its response to different phases
    # we provide the option to focus our analysis on either or both phases of the study.
    

    if(retain_phases == "1"):
        entries = [e for e in entries if "Phase1" in e ]
        print("retained phase 1")
    elif(retain_phases == "2"):
        entries = [e for e in entries if "Phase2" in e ]
        print("retained phase 2")
    elif(retain_phases == "Both"):
        print("Retaining patients that are included in phases 1 & 2")
        phase_1_ids = [p.split(".")[1] for p in  entries if "Phase1" in p]
        phase_2_ids = [p.split(".")[1] for p in  entries if "Phase2" in p]
        # Find the entries that match with both Phase 1 and Phase 2
        common_ids = set(phase_1_ids) & set(phase_2_ids) # set intersection
        entries_matching_both_phases = [entry for entry in entries if any(f".{common_id}." in entry for common_id in common_ids)]
        entries = entries_matching_both_phases
    elif(retain_phases == None):
        print("not applying any filtering over phases")
    else:
        print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts



    # if we want a smaller dataset for testing purposes
    if(subsample is not None):
        entries = entries[0:subsample]


    # We can decide to only include patient who completed a given quantities of timepoints
    # The following stategy for filtering also filters out every patient who have missed a visit up to the given timepoint.
    # This comportement could be tweaked easely later on
    # a bit clunky though



    ########################################################################################
    # there is a bit of trouble shooting left to do in this section, in the case time series + BL
    ########################################################################################
    if(minimum_time_point == "BL" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in BL_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V02" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line to month 6 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        common_ids = set(BL_ids) & set(V02_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V04" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line to month 12 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V06" and as_time_series == False):
        print("retaining all patient who have at least passed the Base Line to month 24 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        V06_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V06"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    
    # if we want time series, we constrain them to only patients that went through every visits.
    elif(minimum_time_point == "V08" or as_time_series == True):
        print("retaining all patient who have passed all visits...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        V06_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V06"] 
        V08_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V08"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) & set(V08_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries


    #### FOR EXERIMENTATION
    if(keep_only_BL):
        entries = [e for e in entries if e.split(".")[2] == "BL"]


    # sanity check : are the patient numbers actually numeric ? 
    entries = [e for e in entries if e.split(".")[1].isnumeric() ]

    # sanity check : don't load patient where some values are missing
    Na_s =  meta_data[meta_data.isna().any(axis=1)]["Patient Number"]
    entries = [e for e in entries if e.split(".")[1] not in str(Na_s) ]


    #### FOR EXERIMENTATION
    if(keep_only_genetic_pd):
        GPD = [1 if ds == "Genetic PD" else 0 for ds in meta_data["Disease Status"] ]
        entries = entries[GPD]



    ###########################################
    ############ loading patients  ############
    ###########################################

    # load the dataset into an array 
    print("loading samples...")
    data = [load_patient_data(os.path.join(path, e)) for e in entries]

    # get the entry name list
    names = get_names(os.path.join(path,entries[0]))

    names = pd.DataFrame([n.split("|") for n in names])
    
    # remove artifacts by keeping samples of correct length
    samples_to_keep = [1 if s.shape == (95309,) else 0 for s in data]

        
    print("loaded",sum(samples_to_keep), "samples")
    
    train_ds = [sample for (sample, test) in  zip(data, samples_to_keep) if test]
    data_array = np.array(train_ds)

    patient_id = [int(p.split(".")[1]) for (p, test) in  zip(entries, samples_to_keep) if test]

    # only keep metadata for selected patients
    meta_data = meta_data.set_index('Patient Number')
    meta_data = meta_data.reindex(index=patient_id)
    meta_data = meta_data.reset_index()





    ###########################################
    ############ feature selection  ###########
    ###########################################
 
    if(feature_selection_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",feature_selection_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data_array, feature_selection_threshold)
        print("removing", len(gene_selected) - sum(gene_selected), "genes under the MAD threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]


    
    print("number of genes selected : ", len(data_array[0]))

    ###########################################
    ############## normalisation  #############
    ###########################################

    
    if(normalization == True): 
        print("normalizing data...")
        data_array = normalize(data_array)

    if(log1p == True): 
        print("log(1 + x) transformation...")
        data_array = np.log1p(data_array)

    # after log1p transform because it already provide us with a very good dataset 
    if(min_max == True):
        print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data_array = scaler.fit_transform(data_array)

    

    ##########################################
    ######## Building the time series ########
    ##########################################
    print("number of seq in the dataset :", len(data_array))

    if(as_time_series == True):
        print("converting samples to time series")
        big_bad_dict = {key: [sample for (sample, name) in zip(data_array, entries) if int(name.split(".")[1]) == key] for key in patient_id}

        print("number of actual individual to be studied :", len(big_bad_dict))

        # Convert dictionary values to numpy arrays
        sequences = np.array(list(big_bad_dict.values()), dtype=np.float32)

        # Step 2: Create a TensorFlow Dataset
        x_train = tf.data.Dataset.from_tensor_slices(sequences)

        if(transpose):
            print("using transposed data...")

            # Define a function to transpose a sequence
            def transpose_sequence(sequence):
                return tf.transpose(sequence, perm=[1, 0])

            # Transpose each element in the dataset
            x_train = x_train.map(transpose_sequence)


        # to keep track of which timeserie correspond to which identifier
        sequence_names = list(big_bad_dict.keys())
    else:
        # we don't assemble the files into timeseries and simply return the TPM values and the corresponding filename
        print("keeping sample as is, no conversion to time series")
        x_train = tf.data.Dataset.from_tensor_slices(data_array)
        sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]

    # make it a batched dataset
    dataset = x_train.batch(batch_size)

    # adding correct attributes
    dataset._name = "transcripts"
    dataset._is_transpose = transpose
    dataset._is_time_series = as_time_series
    
    return dataset, sequence_names, len(data_array[0]), names





### now we design a function that return a dataset of multivriate time series or the individual timestamps
def generate_dataset_cancer(
        path = absolute_path_cancer, 
        feature_selection_threshold = None, 
        batch_size = 64, 
        subsample = None, 
        normalization = True,
        transpose = False,
        MT_removal = True,
        log1p = True,
        min_max = True):




    # getting entries ready

    entries = os.listdir(path)
    # entries are contained into their own subdir
    entries = [[path+"/"+entry+"/"+file for file in os.listdir(path+"/"+entry)] for entry in entries if os.path.isdir(path+"/"+entry)]
    entries = [[e for e in entries if ".tsv" in e ][0] for entries in entries]
    entries = [e for e in entries if "augmented_star_gene_counts" in e ]



    # we load metadata, so we can have access to additional information not included in the filename
    #meta_data = pd.read_excel(metadata_path, header = 1, usecols = range(1,10) )

    ###########################################
    ###### pre-loading patient selection ######
    ###########################################
    # selecting which entires to include in our analysis


    # if we want a smaller dataset for testing purposes
    if(subsample is not None):
        entries = entries[0:subsample]

    # sanity check : don't load patient where some values are missing
    #Na_s =  meta_data[meta_data.isna().any(axis=1)]["Patient Number"]
    #entries = [e for e in entries if e.split(".")[1] not in str(Na_s) ]


    ###########################################
    ############ loading patients  ############
    ###########################################

    # load the dataset into an array 
    print("loading samples...")
    data = [load_patient_data(e, header = 5) for e in entries]

    # get the entry name list
    names = get_names(entries[0], header = 5)
    
    # remove artifacts by keeping samples of correct length
    samples_to_keep = [1 if s.shape == (60660,) else 0 for s in data]   
    print("loaded",sum(samples_to_keep), "/",len(samples_to_keep), "samples")
    
    train_ds = [sample for (sample, test) in  zip(data, samples_to_keep) if test]
    data_array = np.array(train_ds)

    #patient_id = [int(p.split(".")[1]) for (p, test) in  zip(entries, samples_to_keep) if test]

    # only keep metadata for selected patients
    #meta_data = meta_data.set_index('Patient Number')
    #meta_data = meta_data.reindex(index=patient_id)
    #meta_data = meta_data.reset_index()





    ###########################################
    ############ feature selection  ###########
    ###########################################
 
    if(feature_selection_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",feature_selection_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data_array, feature_selection_threshold)
        print("removing", len(gene_selected) - sum(gene_selected), "genes under the MAD threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]


    
    print("number of genes selected : ", len(data_array[0]))
    print("number of genes selected : ", len(names))

    ###########################################
    ############## normalisation  #############
    ###########################################

    
    if(normalization == True): 
        print("normalizing data...")
        data_array = normalize(data_array)

    if(log1p == True): 
        print("log(1 + x) transformation...")
        data_array = np.log1p(data_array)

    # after log1p transform because it already provide us with a very good dataset 
    if(min_max == True):
        print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data_array = scaler.fit_transform(data_array)

    


    print("shape of the dataset :", data_array.shape)

    print("number of seq in the dataset :", len(data_array))

    sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]


    #return data_array, sequence_names, len(data_array[0]), names

    x_train = tf.data.Dataset.from_tensor_slices(data_array)

    # make it a batched dataset
    dataset = x_train.batch(batch_size)

    # adding correct attributes
    dataset._name = "cancer"
    dataset._is_transpose = transpose
    dataset._is_time_series = False
    
    return dataset, sequence_names, len(data_array[0]), names



