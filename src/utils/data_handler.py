from .. import config
from . import feature_selection


import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.cluster.hierarchy as sch
import json

from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler


# for translation of gene symbols
import mygene
mg = mygene.MyGeneInfo()





# default path of the folder containing the salmon files
PPMI_DATA_PATH      = config["PPMI_DATA_PATH"]
PPMI_METADATA_PATH  = config["PPMI_METADATA_PATH"]

CANCER_DATA_PATH    = config["CANCER_DATA_PATH"]

BRCA_DATA_PATH      = config["BRCA_DATA_PATH"]
BRCA_METADATA_PATH  = config["BRCA_METADATA_PATH"]
BRCA_SUBTYPES_PATH  = config["BRCA_SUBTYPES_PATH"]




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
        path = PPMI_DATA_PATH, 
        metadata_path = PPMI_METADATA_PATH,
        MAD_threshold = None, 
        subsample = None, 
        retain_phases = None,
        feature_selection_proceedure = None,
        sgdc_params = None,
        class_balancing = None,
        normalization = False,
        minimum_time_point = "BL",
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
    if(minimum_time_point == "BL"):
        print("retaining all patient who have at least passed the Base Line Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in BL_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V02"):
        print("retaining all patient who have at least passed the Base Line to month 6 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        common_ids = set(BL_ids) & set(V02_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V04" ):
        print("retaining all patient who have at least passed the Base Line to month 12 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    elif(minimum_time_point == "V06"):
        print("retaining all patient who have at least passed the Base Line to month 24 Visit...")
        BL_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "BL"] 
        V02_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V02"] 
        V04_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V04"] 
        V06_ids = [p.split(".")[1] for p in  entries if p.split(".")[2] == "V06"] 
        common_ids = set(BL_ids) & set(V02_ids) & set(V04_ids) & set(V06_ids) 
        matchin_entries = [entry for entry in entries if entry.split(".")[1] in common_ids]
        entries = matchin_entries
    # if we want time series, we constrain them to only patients that went through every visits.
    elif(minimum_time_point == "V08"):
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


    # we don't assemble the files into timeseries and simply return the TPM values and the corresponding filename
    print("keeping sample as is, no conversion to time series")
    dataset = data_array
    sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]

    


    metadata = {"name" : "genes",
                "feature_names" : query_result,
                "seq_names" : sequence_names,
                "n_features" : len(data_array[0]),
                "n_seq" : len(sequence_names)} 

    return dataset, metadata
    












### now we design a function that return a dataset of multivriate time series or the individual timestamps
def generate_dataset_transcripts(
        path = PPMI_DATA_PATH, 
        metadata_path = PPMI_METADATA_PATH,
        MAD_threshold = None, 
        subsample = None, 
        retain_phases = None,
        normalization = False,
        time_point = "BL",
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




    if(time_point == "BL"):
        print("retaining only the Base Line Visit data...")
        entries = [e for e in  entries if e.split(".")[2] == "BL"] 
    
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
    names = get_names(os.path.join(path,entries[0])).iloc[:,0]
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
        print(names)
        print(gene_selected)
        names = names[gene_selected]


    
    print("number of genes selected : ", len(data[0]))

    ###########################################
    ############## normalisation  #############
    ###########################################

    
    if normalization: 
        print("normalizing data...")
        data = normalize(data)

    if log1p : 
        print("log(1 + x) transformation...")
        data = np.log1p(data)

    # after log1p transform because it already provide us with a very good dataset 
    if min_max:
        print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data = scaler.fit_transform(data)

    


    print("number of seq in the dataset :", len(data))



    sequence_names = [f for (f, test) in  zip(entries, samples_to_keep) if test]


    metadata = {"name" : "transcripts",
                "feature_names" : names,
                "sequence_names" : sequence_names,
                "n_features" : len(data[0]),
                "meta_data" : meta_data,
                "subtypes" : meta_data["Cohort"]} 

    return data, metadata
    
















### now we design a function that return a dataset of multivriate time series or the individual timestamps
def generate_dataset_BRCA(
        path = BRCA_DATA_PATH, 
        metadata_path = BRCA_METADATA_PATH,
        subtypes_table = BRCA_SUBTYPES_PATH,
        MAD_threshold = None, 
        LS_threshold = None, 
        expression_threshold = None, 
        subsample = None, 
        normalization = False,
        MT_removal = False,
        log1p = True,
        min_max = True,
        keep_only_protein_coding = False,
        sort = True,
        verbose = 1):

    # getting entries ready

    entries = os.listdir(path)
    # entries are contained into their own subdir
    entries = [[path+"/"+entry+"/"+file for file in os.listdir(path+"/"+entry)] for entry in entries if os.path.isdir(path+"/"+entry)]
    entries = [[e for e in entries if ".tsv" in e ][0] for entries in entries]
    entries = [e for e in entries if "augmented_star_gene_counts" in e ]



    # we load metadata, so we can have access to additional information not included in the filename
    f = open(metadata_path)
    meta_data = json.load(f)
    
    ###########################################
    ###### pre-loading patient selection ######
    ###########################################
    # selecting which entires to include in our analysis


    # if we want a smaller dataset for testing purposes
    if(subsample is not None):
        entries = entries[0:subsample]




    ###########################################
    ############ loading patients  ############
    ###########################################

    # load the dataset into an array 
    if verbose:
        print("loading samples...")
    data = [load_patient_data(e, header = 5) for e in entries]

    # get the entry name list    
    names = pd.DataFrame(get_names(entries[0], header = 1, skiprows = [2,3,4,5]))

    data_array = np.array(data)




    ###########################################
    ################ subtypes  ################
    ###########################################


    

    # using chatGPT overlord to solve this 


    # Step 1: Construct a mapping from file_name to entity_submitter_id
    file_to_id = {}
    for item in meta_data:
        file_name = item["file_name"]
        entity_id = item["associated_entities"][0]["entity_submitter_id"]
        file_to_id[file_name] = entity_id

    # Step 2: For each file name in metadata["sequence_names"], find its corresponding entity_submitter_id
    ids = []
    for file_name in entries:
        # Extract the last part of the path which corresponds to the file name
        last_part = file_name.split('/')[-1]
        if last_part in file_to_id:
            ids.append(file_to_id[last_part])
        else:
            ids.append(None)  # or some default value indicating no match found


    subtypes_table = pd.read_csv(subtypes_table, index_col= 0)

    subtypes_dict = {str(index)[:12]: subtype for index, subtype in subtypes_table.itertuples()}
    subtypes = [subtypes_dict.get(identifier[:12], None) for identifier in ids]


    ###########################################
    ############ feature selection  ###########
    ###########################################

    if(MT_removal):
        gene_selected =  [False if name.startswith("MT-") else True for name in names["gene_name"]]
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "mithocondrial genes from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(keep_only_protein_coding):
        gene_selected = names["gene_type"] == "protein_coding"
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "Non coding genes from dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(expression_threshold is not None):
        if verbose:
            print("selecting genes based on expression threshold: ",expression_threshold, "...")
        gene_selected = feature_selection.expression_selection(data_array, expression_threshold, verbose)
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "genes under the expression threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(MAD_threshold is not None):
        if verbose:
            print("selecting genes based on median absolute deviation (MAD) threshold: ",MAD_threshold, "...")
        gene_selected = feature_selection.MAD_selection(data_array, MAD_threshold, verbose)
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "genes under the MAD threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(LS_threshold is not None):
        if verbose:
            print("selecting genes based on Laplacian Score (LS) threshold: ",LS_threshold, "...")
        gene_selected = feature_selection.LS_selection(data_array, LS_threshold, 5, verbose)
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "genes under the LS threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]




    if verbose:
        print("number of genes selected : ", len(data_array[0]))
        print("matching : ", len(names))
        

    
    ###########################################
    ############## Sorting genes  #############
    ###########################################
    if(sort):
        print("retriving symbols for genes")
        queries = [name.split(".")[0] for name in names["gene_id"]]
        query_result = mg.querymany(queries, fields = ['genomic_pos', 'symbol'], scopes='ensembl.gene', species='human', verbose = False, as_dataframe = True)

        query_result = query_result.reset_index()
        query_result = query_result.drop_duplicates(subset = ["query"])
        # here we have the correct length

        names_ = [q if(pd.isna(s)) else s for (s,q) in zip(query_result["symbol"],query_result["query"])]
        query_result['name'] = names_

        print("sorting based on genomic position chr then transcript start...")
        # reset the indexes because of all the previous transformations we have done
        query_result = query_result.reset_index(drop=True)
        query_result = query_result.sort_values(['genomic_pos.chr', 'genomic_pos.start'], ascending=[True, True])
        # Extract the sorted rows as a NumPy array
        data_array = data_array[:, query_result.index]
        names = names.iloc[query_result.index]






    ###########################################
    ############## normalisation  #############
    ###########################################

    


    if(log1p == True): 
        if verbose:
            print("log(1 + x) transformation...")
        data_array = np.log1p(data_array)

    # after log1p transform because it already provide us with a very good dataset 
    if(min_max == True):
        if verbose:
            print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data_array = scaler.fit_transform(data_array)

    if(normalization == True): 
        if verbose:
            print("normalizing data...")
        scaler = StandardScaler()
        data_array = scaler.fit_transform(data_array)


    if verbose:
        print("shape of the dataset :", data_array.shape)
        print("number of seq in the dataset :", len(data_array))


    metadata = {"name"           : "cancer",
                "feature_names"  : names,
                "sequence_names" : entries,
                "n_features"     : len(data_array[0]),
                "subtypes"       : subtypes}
    
        

    return data_array, metadata

