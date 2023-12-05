from .. import config
from . import feature_selection


import os
import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler

from typing import Optional, List


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



# Helper functions 

def load_patient_data(filename: str, header: int = 0) -> pd.Series:
    """
    Load patient data from a given file.

    Args:
    filename (str): The path to the file to be loaded.
    header (int, optional): The row number to use as the column names. Defaults to 0.

    Returns:
    pd.Series: A pandas Series containing TPM values from the file.
    """
    try:
        data = pd.read_table(filename, header=header)
        return data.iloc[:, 3]
    except FileNotFoundError:
        print(f"File not found: {filename}")

def get_gene_names_from_file(filename: str, header: int = 0, skiprows: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Retrieves a list of gene names from a given file.

    Args:
    filename (str): Path to the file from which to read the names.
    header (int, optional): Row number to use as the header (column names). Defaults to 0.
    skiprows (list of int, optional): Line numbers to skip while reading the file. Defaults to None.

    Returns:
    pd.DataFrame: A DataFrame containing the names from the file.
    """
    try:
        names = pd.read_table(filename, header=header, skiprows=skiprows)
        return names
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except pd.errors.ParserError as e:
        print(f"Error parsing file {filename}: {e}")



def load_metadata(metadata_path, columns=None):
    """Load metadata from a given path."""
    return pd.read_excel(metadata_path, header=1, usecols=columns if columns else range(1, 10))

def filter_entries_by_phase(entries, retain_phases):
    """Filter entries based on phase criteria."""
    if retain_phases == "1":
        return [e for e in entries if "Phase1" in e]
    elif retain_phases == "2":
        return [e for e in entries if "Phase2" in e]
    elif(retain_phases == "Both"):
        print("Retaining patients that are included in phases 1 & 2")
        phase_1_ids = [p.split(".")[1] for p in  entries if "Phase1" in p]
        phase_2_ids = [p.split(".")[1] for p in  entries if "Phase2" in p]
        # Find the entries that match with both Phase 1 and Phase 2
        common_ids = set(phase_1_ids) & set(phase_2_ids) # set intersection
        return [entry for entry in entries if any(f".{common_id}." in entry for common_id in common_ids)]
    elif(retain_phases == None):
        print("not applying any filtering over phases")
        return entries
    else:
        print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts
        return entries


def generate_dataset(dataset_type, path = PPMI_DATA_PATH , metadata_path = PPMI_METADATA_PATH, 
                     subsample=None, retain_phases=None, MAD_threshold=None, LS_threshold=None, 
                     expression_threshold=None, normalization=False, time_points=["BL"], 
                     MT_removal=False, log1p=True, min_max=True, keep_only_symbols=False, 
                     drop_ambiguous_pos=False, sort_symbols=False, select_subtypes=None, 
                     verbose=0):
    
    """
    Generate a genomic or transcriptomic dataset from specified files, applying various
    processing and filtering steps.

    Args:
        dataset_type (str): Specifies the type of dataset to generate. Options are 'genomic' or 'transcriptomic'.
        path (str): Filepath to the directory containing the data files.
        metadata_path (str): Filepath to the metadata file.
        subsample (int, optional): If specified, limits the number of samples to process.
        retain_phases (str, optional): Specifies which study phases to include in the analysis (e.g., '1', '2', 'Both').
        MAD_threshold (float, optional): Threshold for Median Absolute Deviation-based feature selection.
        LS_threshold (float, optional): Threshold for Laplacian Score-based feature selection.
        expression_threshold (float, optional): Threshold for expression level-based gene selection.
        normalization (bool, optional): If True, normalizes the data.
        time_points (list of str, optional): Specific time points to include in the analysis.
        MT_removal (bool, optional): If True, removes mitochondrial genes from the dataset.
        log1p (bool, optional): If True, applies log(1 + x) transformation to the data.
        min_max (bool, optional): If True, scales the data using Min-Max normalization.
        keep_only_symbols (bool, optional): For genomic data, if True, retains only symbol-named genes.
        drop_ambiguous_pos (bool, optional): For genomic data, if True, drops genes with ambiguous genomic positions.
        sort_symbols (bool, optional): For genomic data, if True, sorts genes based on their genomic position.
        select_subtypes (list of str, optional): Filters the data to include only specified subtypes.
        gene_selection_file (str, optional): Filepath to a gene selection file.
        verbose (int, optional): Verbosity level for output messages.

    Returns:
        tuple: A tuple containing:
            - data_array (numpy.ndarray): The processed data array.
            - metadata (dict): A dictionary containing metadata information about the dataset.

    Raises:
        ValueError: If an invalid dataset type is provided.
    
    The function applies a series of data preprocessing, normalization, and feature selection 
    steps to generate a dataset ready for analysis. It is capable of handling both genomic 
    and transcriptomic data, contingent on the specified 'dataset_type'. The function also 
    accommodates various filtering and data transformation options.
    """

    # Common code for loading and filtering entries
    entries = os.listdir(path)
    if dataset_type == 'genomic':
        dataset_of_interest = "genes"

    elif dataset_type == 'transcriptomic':
        dataset_of_interest = "transcripts"
    else:
        raise ValueError("Invalid dataset type. Choose 'genomic' or 'transcriptomic'.")

    entries = [e for e in entries if dataset_of_interest in e]
    entries.sort()

    meta_data = pd.read_excel(metadata_path, header=1, usecols=range(1, 10))

    if subsample is not None:
        entries = entries[:subsample]

    entries = filter_entries_by_phase(entries, retain_phases)

    if(time_points is not None):
        print("retaining all patient who have at least passed the", time_points,"Visit...")
        entries = [p for p in  entries if p.split(".")[2] in time_points] 

    # sanity check : are the patient numbers actually numeric ? 
    entries = [e for e in entries if e.split(".")[1].isnumeric() ]

    # sanity check : don't load patient where some values are missing
    Na_s =  meta_data[meta_data.isna().any(axis=1)]["Patient Number"]
    entries = [e for e in entries if e.split(".")[1] not in str(Na_s) ]

    if(select_subtypes is not None):
        meta_data = meta_data[meta_data["Disease Status"].isin(select_subtypes)]
        id_pd = meta_data["Patient Number"].tolist()
        entries = [e for e in entries if int(e.split(".")[1]) in id_pd]

    ###########################################
    ############ loading patients  ############
    ###########################################

    # load the dataset into an array 
    print("loading samples...")    
    data_array = np.array([load_patient_data(os.path.join(path, e)) for e in entries])
    print("loaded ",len(data_array), "samples")

    patient_id = [int(p.split(".")[1]) for p in entries]

    # only keep metadata for selected patients
    meta_data = meta_data.set_index('Patient Number')
    meta_data = meta_data.reindex(index=patient_id)
    meta_data = meta_data.reset_index()

    ###########################################
    ############ feature selection  ###########
    ###########################################

 



    # Load data and perform dataset-specific processing
    if dataset_type == 'genomic':
        # get the genes name list
        names = get_gene_names_from_file(os.path.join(path,entries[0])).iloc[:,0]
        # formatting gene names to get rid of the version number
        names = [n.split(".")[0] for n in names]
        print("retriving symbols for genes")
        query_result = mg.querymany(names, fields = ['genomic_pos', 'symbol'], scopes='ensembl.gene', species='human', verbose = False, as_dataframe = True)
        query_result = query_result.reset_index()
        query_result = query_result.drop_duplicates(subset = ["query"])
        # here we have the correct length
        names = [q if(pd.isna(s)) else s for (s,q) in zip(query_result["symbol"],query_result["query"])]
        query_result['name'] = names

    elif dataset_type == 'transcriptomic':
        # get the entry name list
        names = get_gene_names_from_file(os.path.join(path,entries[0])).iloc[:,0]
        query_result = pd.DataFrame([n.split("|") for n in names])
        query_result = query_result.set_axis([
            #'trascript_id',  # we turn transcript_id into name for cross compatibility
            'name', 
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

    else:
        raise ValueError("Invalid dataset type. Choose 'genomic' or 'transcriptomic'.")




    if(MT_removal == True):
        gene_selected = [False if q.startswith('MT') else True for q in query_result['name']]
        print("removing", len(gene_selected) - sum(gene_selected), "mithocondrial genes from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(keep_only_symbols == True and dataset_type == 'genomic'):
        gene_selected = [False if s.startswith('ENSG') else True for s in query_result['name']]
        print("removing", len(gene_selected) - sum(gene_selected), "not found symbols from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(drop_ambiguous_pos == True and dataset_type == 'genomic'):
        gene_selected = query_result["genomic_pos"].isna()
        print("removing", len(gene_selected) - sum(gene_selected), "ambigously positioned symbols from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(expression_threshold is not None):
        if verbose:
            print("selecting genes based on expression threshold: ",expression_threshold, "...")
        gene_selected = feature_selection.expression_selection(data_array, expression_threshold, verbose)
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "genes under the expression threshold from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(MAD_threshold is not None):
        MAD_ceiling = 150
        print("selecting genes based on median absolute deviation window: [",MAD_threshold,",", MAD_ceiling, "] ...")
        gene_selected = feature_selection.MAD_selection(data_array, MAD_threshold, verbose = verbose)
        print("removing", len(gene_selected) - sum(gene_selected), "genes out of the MAD window from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    if(LS_threshold is not None):
        if verbose:
            print("selecting genes based on Laplacian Score (LS) threshold: ",LS_threshold, "...")
        gene_selected = feature_selection.LS_selection(data_array, LS_threshold, 5, verbose)
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "genes under the LS threshold from the dataset")
        data_array = data_array[:,gene_selected]
        query_result = query_result[gene_selected]

    print("number of genes selected : ", len(data_array[0]))

    ###########################################
    ################# sorting  ################
    ###########################################

    if(sort_symbols and dataset_type == 'genomic'):
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


    print("number of seq in the dataset :", len(data_array))


    metadata = {"name" : dataset_of_interest,
                "feature_names" : query_result,
                "seq_names" : entries,
                "n_features" : len(data_array[0]),
                "n_seq" : len(entries),
                "meta_data" : meta_data,
                "subtypes" : meta_data["Disease Status"]} 

    return data_array, metadata


















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
    names = pd.DataFrame(get_gene_names_from_file(entries[0], header = 1, skiprows = [2,3,4,5]))

    data_array = np.array(data)




    ###########################################
    ################ subtypes  ################
    ###########################################


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
        MAD_ceiling = 100

        if verbose:
            print("selecting genes based on median absolute deviation window: [",MAD_threshold,",", MAD_ceiling, "] ...")
        gene_selected = feature_selection.MAD_selection(data_array, MAD_threshold, MAD_ceiling, verbose)
        if verbose:
            print("removing", len(gene_selected) - sum(gene_selected), "genes out of the MAD window from the dataset")
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

