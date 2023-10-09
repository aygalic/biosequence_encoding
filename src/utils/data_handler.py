import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import scipy
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler


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

absolute_path_BRCA = '/Users/aygalic/Thesis/data/BRCA'  



# from filename to tensor
# here we open a single file passed as "filename" we return a tensor of TPM values.
def load_patient_data(filename, header = 0):
    data = pd.read_table(filename, header = header)
    return data.iloc[:, 3]

# here we open a single file passed as "filename" we return a lit of the values names.
def get_names(filename, header = 0, skiprows= None):
    names = pd.read_table(filename, header = header, skiprows = skiprows)
    return names#.iloc[:, 0]


### now we design a function that return a dataset of multivriate time series or cell wise observations
def generate_dataset_genes(
        path = absolute_path, 
        metadata_path = metadata_path,
        MAD_threshold = None, 
        batch_size = 64, 
        subsample = None, 
        retain_phases = None,
        feature_selection_proceedure = None,
        sgdc_params = None,
        class_balancing = None,
        normalization = False,
        minimum_time_point = "BL",
        as_time_series = False,
        transpose = False,
        MT_removal = False,
        log1p = True,
        min_max = True,
        keep_only_symbols = False,
        drop_ambiguous_pos = False,
        sort_symbols = False,
        gene_selection_file = None):

    dataset_of_interest = "genes"

    # getting entries ready
    # each couple of entries correspond to one patient, we are only interested in the "transcript" files
    entries = os.listdir(path)
    #entries_transcripts = [e for e in entries if "transcripts" in e ]
    entries = [e for e in entries if dataset_of_interest in e ]
    entries.sort()

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


    # sanity check : are the patient numbers actually numeric ? 
    entries = [e for e in entries if e.split(".")[1].isnumeric() ]

    # sanity check : don't load patient where some values are missing
    Na_s =  meta_data[meta_data.isna().any(axis=1)]["Patient Number"]
    entries = [e for e in entries if e.split(".")[1] not in str(Na_s) ]



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
    
    data_array = np.array([sample for (sample, test) in  zip(data, samples_to_keep) if test])

    patient_id = [int(p.split(".")[1]) for (p, test) in  zip(entries, samples_to_keep) if test]

    # only keep metadata for selected patients
    meta_data = meta_data.set_index('Patient Number')
    meta_data = meta_data.reindex(index=patient_id)
    meta_data = meta_data.reset_index()

    ###########################################
    ############ feature selection  ###########
    ###########################################

    if(gene_selection_file is not None):
        names = pd.Series(names)
        suggested_genes = pd.read_csv(gene_selection_file, sep='\t')
        suggested_genes = suggested_genes.rename(columns={'Unnamed: 0': 'gene'})
        gene_selected = names.isin(suggested_genes["gene"])
        print("number of genes selected:", sum(gene_selected))
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]


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

    if(MAD_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",MAD_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data_array, MAD_threshold)
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

    if as_time_series:
        print("converting samples to time series")
        time_series_dict = {key: [sample for (sample, name) in zip(data_array, entries) if int(name.split(".")[1]) == key] for key in patient_id}

        print("number of actual individuals to be studied:", len(time_series_dict))

        # Convert dictionary values to NumPy arrays
        sequences = np.array(list(time_series_dict.values()), dtype=np.float32)

        if transpose:
            print("using transposed data...")

            # Define a function to transpose a sequence
            def transpose_sequence(sequence):
                return np.transpose(sequence, axes=[1, 0])

            # Transpose each element in the sequences array
            dataset = np.array([transpose_sequence(sequence) for sequence in sequences], dtype=np.float32)

        # To keep track of which time series correspond to which identifier
        sequence_names = list(time_series_dict.keys())
    else:
        # we don't assemble the files into timeseries and simply return the TPM values and the corresponding filename
        print("keeping sample as is, no conversion to time series")
        dataset = data_array
        sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]

    


    metadata = {"name" : "genes",
                "is_transpose": transpose,
                "is_time_series" : as_time_series,
                "feature_names" : query_result,
                "seq_names" : sequence_names,
                "n_features" : len(data_array[0]),
                "n_seq" : len(sequence_names)} 

    return dataset, metadata
    












### now we design a function that return a dataset of multivriate time series or the individual timestamps
def generate_dataset_transcripts(path = absolute_path, 
                     metadata_path = metadata_path,
                     MAD_threshold = None, 
                     batch_size = 64, 
                     subsample = None, 
                     retain_phases = None,
                     normalization = False,
                     minimum_time_point = "BL",
                     as_time_series = False,
                     transpose = False,
                     MT_removal = False,
                     log1p = True,
                     min_max = True,
                     gene_selection_file = None):

    # getting entries ready
    # each couple of entries correspond to one patient, we are only interested in the "transcript" files
    entries = os.listdir(path)
    #entries_transcripts = [e for e in entries if "transcripts" in e ]
    entries = [e for e in entries if "transcripts" in e ]
    entries.sort()

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

    # sanity check : are the patient numbers actually numeric ? 
    entries = [e for e in entries if e.split(".")[1].isnumeric() ]

    # sanity check : don't load patient where some values are missing
    Na_s =  meta_data[meta_data.isna().any(axis=1)]["Patient Number"]
    entries = [e for e in entries if e.split(".")[1] not in str(Na_s) ]





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
    
    data = [sample for (sample, test) in  zip(data, samples_to_keep) if test]
    data = np.array(data)

    patient_id = [int(p.split(".")[1]) for (p, test) in  zip(entries, samples_to_keep) if test]

    # only keep metadata for selected patients
    meta_data = meta_data.set_index('Patient Number')
    meta_data = meta_data.reindex(index=patient_id)
    meta_data = meta_data.reset_index()





    ###########################################
    ############ feature selection  ###########
    ###########################################
 
    if(gene_selection_file is not None):
        name_df = names.set_axis([
            'trascript_id', 
            'gene_id', 
            'idk', 
            'idk', 
            'transcript_variant', 
            'symbol', 
            'length', 
            'untranslated_region_3', 
            'coding_region', 
            'untranslated_region_5', 
            'idk'], axis=1)
        suggested_genes = pd.read_csv(gene_selection_file, sep='\t')
        suggested_genes = suggested_genes.rename(columns={'Unnamed: 0': 'gene_id'})
        mask_gene_name = name_df["symbol"].isin(suggested_genes["name"])
        mask_gene_id = pd.Series([id.split(".")[0] for id in name_df["gene_id"]]).isin(suggested_genes["gene_id"])
        gene_selected = mask_gene_name | mask_gene_id
        data = data[:,gene_selected]
        names = names[gene_selected]


    

    if(MAD_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",MAD_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data, MAD_threshold)
        print("removing", len(gene_selected) - sum(gene_selected), "genes under the MAD threshold from the dataset")
        data = data[:,gene_selected]
        names = names[gene_selected]


    
    print("number of genes selected : ", len(data[0]))

    ###########################################
    ############## normalisation  #############
    ###########################################

    
    if(normalization == True): 
        print("normalizing data...")
        data = normalize(data)

    if(log1p == True): 
        print("log(1 + x) transformation...")
        data = np.log1p(data)

    # after log1p transform because it already provide us with a very good dataset 
    if(min_max == True):
        print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data = scaler.fit_transform(data)

    

    ##########################################
    ######## Building the time series ########
    ##########################################
    print("number of seq in the dataset :", len(data))

    if as_time_series:
        print("converting samples to time series")
        time_series_dict = {key: [sample for (sample, name) in zip(data, entries) if int(name.split(".")[1]) == key] for key in patient_id}

        print("number of actual individuals to be studied:", len(time_series_dict))

        # Convert dictionary values to NumPy arrays
        sequences = np.array(list(time_series_dict.values()), dtype=np.float32)

        if transpose:
            print("using transposed data...")

            # Define a function to transpose a sequence
            def transpose_sequence(sequence):
                return np.transpose(sequence, axes=[1, 0])

            # Transpose each element in the sequences array
            data = np.array([transpose_sequence(sequence) for sequence in sequences], dtype=np.float32)

        # To keep track of which time series correspond to which identifier
        sequence_names = list(time_series_dict.keys())
    else:
        # we don't assemble the files into timeseries and simply return the TPM values and the corresponding filename
        print("keeping sample as is, no conversion to time series")
        sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]


    
    metadata = {"name" : "transcripts",
                "is_transpose": transpose,
                "is_time_series" : as_time_series,
                "feature_names" : names,
                "sequence_names" : sequence_names,
                "n_features" : len(data[0])} 

    return data, metadata
    













### now we design a function that return a dataset of multivriate time series or the individual timestamps
def generate_dataset_cancer(
        path = absolute_path_cancer, 
        MAD_threshold = None, 
        batch_size = 64, 
        subsample = None, 
        normalization = False,
        transpose = False,
        MT_removal = False,
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
 
    if(MAD_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",MAD_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data_array, MAD_threshold)
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


    dataset = data_array

    metadata = {"name" : "cancer",
                "is_transpose": transpose,
                "is_time_series" : False,
                "feature_names" : names,
                "sequence_names" : sequence_names,
                "n_features" : len(data_array[0])} 

    return dataset, metadata



### now we design a function that return a dataset of multivriate time series or the individual timestamps
def generate_dataset_BRCA(
        path = absolute_path_BRCA, 
        MAD_threshold = None, 
        var_threshold = None, 
        expression_threshold = None, 
        subsample = None, 
        normalization = False,
        transpose = False,
        MT_removal = False,
        log1p = True,
        min_max = True,
        PAM50 = True,
        keep_only_protein_coding = False,
        verbose = 0):

    # list of PAM50 genes:
    PAM50_genes = [
        "FOXC1", "MIA", "NDC80", "CEP55", "ANLN", "MELK", "GPR160", "TMEM45B", "ESR1", "FOXA1",
        "ERBB2", "GRB7", "FGFR4", "BLVRA", "BAG1", "CDC20", "CCNE1", "ACTR3B", "MYC", "SFRP1",
        "KRT14", "KRT17", "KRT5", "MLPH", "CCNB1", "CDC6", "TYMS", "UBE2T", "RRM2", "MMP11", 
        "CXXC5", "ORC6", "MDM2", "KIF2C",  "PGR", "MKI67", "BCL2", "EGFR", "PHGDH", "CDH3",
        "NAT1", "SLC39A6", "MAPT", "UBE2C", "PTTG1", "EXO1", "CENPF", "NUF2", "MYBL2", "BIRC5"]

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
    names = pd.DataFrame(get_names(entries[0], header = 1, skiprows = [2,3,4,5]))



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
    ################# PAM50  ##################
    ###########################################
    
    if(PAM50):
        # Now let's reproduce the PAM50 algorihtm
        PAM50_mask = [True if name.split(".")[0] in PAM50_genes else False for name in names["gene_name"]]
        print(len(PAM50_mask))
        print(sum(PAM50_mask))
        X = data_array[:,PAM50_mask]

        median_centered_data = X - np.median(X, axis=1)[:, np.newaxis]
        correlation_matrix = np.corrcoef(median_centered_data)

        # Step 3: Perform Hierarchical Clustering with Average Linkage
        linkage_matrix = sch.linkage(correlation_matrix, method='average')

        # Step 4: Cut the Dendrogram into 5 Clusters
        k = 5  # Number of clusters
        cluster_labels = sch.fcluster(linkage_matrix, k, criterion='maxclust')




    ###########################################
    ############ feature selection  ###########
    ###########################################

    if(MT_removal):
        gene_selected =  [False if name.startswith("MT-") else True for name in names["gene_name"]]
        print("removing", len(gene_selected) - sum(gene_selected), "mithocondrial genes from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(keep_only_protein_coding):
        gene_selected = names["gene_type"] == "protein_coding"
        print("removing", len(gene_selected) - sum(gene_selected), "Non coding genes from dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(expression_threshold is not None):
        print("selecting genes based on expression threshold: ",expression_threshold, "...")
        gene_selected = feature_selection.expression_selection(data_array, expression_threshold, verbose)
        print("removing", len(gene_selected) - sum(gene_selected), "genes under the expression threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(MAD_threshold is not None):
        print("selecting genes based on median absolute deviation threshold: ",MAD_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data_array, MAD_threshold, verbose)
        print("removing", len(gene_selected) - sum(gene_selected), "genes under the MAD threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]




    
    print("number of genes selected : ", len(data_array[0]))
    print("matching : ", len(names))

    ###########################################
    ############## normalisation  #############
    ###########################################

    


    if(log1p == True): 
        print("log(1 + x) transformation...")
        data_array = np.log1p(data_array)

    # after log1p transform because it already provide us with a very good dataset 
    if(min_max == True):
        print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data_array = scaler.fit_transform(data_array)

    if(normalization == True): 
        print("normalizing data...")
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)


    print("shape of the dataset :", data_array.shape)

    print("number of seq in the dataset :", len(data_array))

    sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]


    dataset = data_array

    metadata = {"name" : "cancer",
                "is_transpose": transpose,
                "is_time_series" : False,
                "feature_names" : names,
                "sequence_names" : sequence_names,
                "n_features" : len(data_array[0])} 
    
    if(PAM50):
        metadata["PAM50_labels"] = cluster_labels

    return dataset, metadata

