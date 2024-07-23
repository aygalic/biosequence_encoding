"""
This module is designed to generate datasets for genomic, transcriptomic, and specialized dataset (such as BRCA). 
It provides functionalities to process and filter data files, perform various feature selections, and format datasets for further analysis.

Functions in this module allow loading of patient data, retrieval of gene names, metadata processing, and dataset-specific processing. 
This includes options for sub-sampling, phase retention, numerical and logical feature selection, as well as data normalization.

Key Functions:
    - load_patient_data: Loads patient data from a file.
    - get_gene_names_from_file: Retrieves gene names from a file.
    - load_metadata: Loads metadata from a specified file.
    - filter_entries_by_phase: Filters file entries based on study phases.
    - retrive_position: Retrieves genomic positions for gene names.
    - generate_dataset: Generates a dataset with specified parameters for genomic, transcriptomic, or BRCA analysis.

The module uses pandas for data handling and sklearn for normalization purposes. It also employs the mygene library for gene symbol translation and genomic position retrieval.

Example:
    To generate a genomic dataset:
        dataset, metadata = generate_dataset('genomic', path='path_to_data', metadata_path='path_to_metadata')

Dependencies:
    - pandas
    - numpy
    - json
    - sklearn.preprocessing
    - mygene
"""

import json
import os
from typing import List, Optional

# for translation of gene symbols
import mygene
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize

from ..utils import feature_selection
from . import BRCA_DATA_PATH, BRCA_METADATA_FILE, BRCA_SUBTYPES_FILE

mg = mygene.MyGeneInfo()





def load_patient_data(filename: str, header: int = 0) -> pd.Series:
    """
    Load patient data from a specified file.

    Args:
        filename (str): Path to the data file.
        header (int, optional): Row number to use as the header. Defaults to 0.

    Returns:
        pd.Series: A pandas Series containing TPM values from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    try:
        data = pd.read_table(filename, header=header)
        return data.iloc[:, 3]
    except FileNotFoundError:
        print(f"File not found: {filename}")


def get_gene_names_from_file(filename: str, header: int = 0, skiprows: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Retrieve a list of gene names from a specified file.

    Args:
        filename (str): Path to the file from which to read the names.
        header (int, optional): Row number to use as the header (column names). Defaults to 0.
        skiprows (list of int, optional): Rows to skip at the start of the file.

    Returns:
        pd.DataFrame: A DataFrame containing the names from the file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ParserError: If there is an error in parsing the file.
    """

    try:
        names = pd.read_table(filename, header=header, skiprows=skiprows)
        return names
    except FileNotFoundError:
        print(f"File not found: {filename}")
    except pd.errors.ParserError as e:
        print(f"Error parsing file {filename}: {e}")


def load_metadata(metadata_path, columns=None):
    """
    Wrapper to load metadata from the given file path.

    Args:
        metadata_path (str): Path to the metadata file.
        columns (list, optional): Specific columns to load from the file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded metadata.
    """
    return pd.read_excel(metadata_path, header=1, usecols=columns if columns else range(1, 10))

def filter_entries_by_phase(entries, retain_phases, verbose = 0):
    """
    Filter file entries based on specified study phases.

    Args:
        entries (list): List of file entries.
        retain_phases (str): Study phases to retain ('1', '2', 'Both', or None).

    Returns:
        list: Filtered list of entries.
    """

    if retain_phases == "1":
        return [e for e in entries if "Phase1" in e]
    elif retain_phases == "2":
        return [e for e in entries if "Phase2" in e]
    elif(retain_phases == "Both"):
        if verbose: print("Retaining patients that are included in phases 1 & 2")
        phase_1_ids = [p.split(".")[1] for p in  entries if "Phase1" in p]
        phase_2_ids = [p.split(".")[1] for p in  entries if "Phase2" in p]
        common_ids = set(phase_1_ids) & set(phase_2_ids) # set intersection
        return [entry for entry in entries if any(f".{common_id}." in entry for common_id in common_ids)]
    elif(retain_phases == None):
        if verbose: print("not applying any filtering over phases")
        return entries
    else:
        if verbose: print("Warning: 'retain_phases' argment wrong.") # couldn't use warning due to conflicts
        return entries

def retrive_position(names, drop_na=False, verbose=0):
    """
    Retrieve genomic positions for a list of gene names.

    Args:
        names (pd.DataFrame): DataFrame containing gene names.
        drop_na (bool, optional): Flag to drop NA values. Defaults to False.
        verbose (int, optional): Verbosity level.

    Returns:
        pd.DataFrame: DataFrame with retrieved genomic positions and symbols.
    """
    if verbose: print("retriving",len(names), "symbols for genes")
    query_result = mg.querymany(names['query'], fields = ['genomic_pos', 'symbol'], scopes='ensembl.gene', species='human', verbose = False, as_dataframe = True)
    query_result = query_result.reset_index()
    if verbose: print("Found",len(query_result), "symbols before duplicate removal")
    query_result = query_result.drop_duplicates(subset = ["query"])
    if verbose: print(len(query_result), "symbols after duplicate removal")
    if drop_na:
        query_result['name'] = [q if(pd.isna(s)) else s for (s,q) in zip(query_result["symbol"],query_result["query"])]
    return query_result



def generate_dataset(
        dataset_type, path = None , metadata_path = None, subtypes_table = None,
        subsample=None, retain_phases=None, MAD_threshold=None, LS_threshold=None,
        expression_threshold=None, normalization=False, time_points=["BL"],
        keep_only_protein_coding = None, log1p=True, min_max=True, keep_only_symbols=False,
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
        verbose (int, optional): Verbosity level.

    Returns:
        tuple: A tuple containing:
            - data_array (numpy.ndarray): The processed data array.
            - metadata (dict): A dictionary containing metadata information about the dataset.
    
    This function processes and prepares a dataset for analysis, including steps like
    loading data, metadata processing, feature selection, and normalization.

    """

    if dataset_type in ["transcriptomic", "genomic"] :
        # Rooting path
        if path is None:
            path = BRCA_DATA_PATH 
            metadata_path = BRCA_METADATA_FILE

        # we load metadata, so we can have access to additional information not included in the filename
        meta_data = pd.read_excel(metadata_path, header=1, usecols=range(1, 10))
        data_array_header = 0
        if dataset_type == 'genomic': dataset_of_interest = "genes"
        elif dataset_type == 'transcriptomic': dataset_of_interest = "transcripts"

    elif dataset_type == 'BRCA':
        if path is None:
            path = BRCA_DATA_PATH
            metadata_path = BRCA_METADATA_FILE
            subtypes_table = BRCA_SUBTYPES_FILE # provided by supervisor

        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)
        data_array_header = 5
        dataset_of_interest = "augmented_star_gene_counts"

    if verbose: print("path:",path)
    
    # Loading entries names 
    entries = os.listdir(path)

    # adding consistent path for each entry 
    if dataset_type in ["transcriptomic", "genomic"] :
        entries = [os.path.join(path, e) for e in entries]
    if dataset_type == 'BRCA':
        # entries are contained into their own subdir.
        
        entries = [entry for entry in entries if os.path.isdir(path / entry)]
        entries = [[path / entry / file for file in os.listdir(path / entry)] for entry in entries]
        entries = [[entry for entry in files if entry.suffix== ".tsv" ] for files in entries]
        entries = [entry[0] for entry in entries]


    # filtering for files that are actually the correct entries by filename
    entries = [e for e in entries if dataset_of_interest in e.stem]
    entries.sort()

    # taking subsample for quicker processing
    if subsample is not None: entries = entries[:subsample]



    ###########################################
    ########### filtering patients  ###########
    ###########################################

    # apply filterring for PPMI dataset
    if dataset_type in ["transcriptomic", "genomic"] :
        entries = filter_entries_by_phase(entries, retain_phases)

        if(time_points is not None):
            if verbose: print("retaining all patient who have at least passed the", time_points,"Visit...")
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
    if verbose: print("loading samples...") 
    data_array = np.array([load_patient_data(e, header = data_array_header) for e in entries])
    if verbose: print("loaded ",len(data_array), "samples")



    ###########################################
    ################ subtypes  ################
    ###########################################

    if dataset_type in ["transcriptomic", "genomic"] :
        patient_id = [int(p.split(".")[1]) for p in entries]
        # only keep metadata for selected patients
        meta_data = meta_data.set_index('Patient Number')
        meta_data = meta_data.reindex(index=patient_id)
        meta_data = meta_data.reset_index()
        subtypes = meta_data["Disease Status"]

    elif dataset_type == "BRCA":
        # Step 1: Construct a mapping from file_name to entity_submitter_id
        file_to_id = {}
        for item in meta_data:
            file_name = item["file_name"][:-4] # remove the .tsv extension
            entity_id = item["associated_entities"][0]["entity_submitter_id"]
            file_to_id[file_name] = entity_id

        # Step 2: For each file name in metadata["sequence_names"], find its corresponding entity_submitter_id
        patient_id = []
        for file_name in entries:
            if file_name.stem in file_to_id.keys():
                patient_id.append(file_to_id[file_name.stem])

        breakpoint()

        subtypes_table = pd.read_csv(subtypes_table, index_col= 0)

        subtypes_dict = {str(index)[:12]: subtype for index, subtype in subtypes_table.itertuples()}
        subtypes = [subtypes_dict.get(identifier[:12], None) for identifier in patient_id]

    ###########################################
    ####### Numerical feature selection  ######
    ###########################################
    
    # Load data and perform dataset-specific processing
    if dataset_type in['genomic','transcriptomic'] :
        names = get_gene_names_from_file(os.path.join(path,entries[0])).iloc[:,0]

    elif dataset_type == 'BRCA':
        names = pd.DataFrame(get_gene_names_from_file(entries[0], header = 1, skiprows = [2,3,4,5]))

    # numerical feature selection

    if(expression_threshold is not None):
        if verbose: print("selecting genes based on expression threshold: ",expression_threshold, "...")
        gene_selected = feature_selection.expression_selection(data_array, expression_threshold, verbose)
        if verbose: print("removing", len(gene_selected) - sum(gene_selected), "genes under the expression threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(MAD_threshold is not None):
        MAD_ceiling = 150
        if verbose: print("selecting genes based on median absolute deviation window: [",MAD_threshold,",", MAD_ceiling, "] ...")
        gene_selected = feature_selection.MAD_selection(data_array, MAD_threshold, verbose = verbose)
        if verbose: print("removing", len(gene_selected) - sum(gene_selected), "genes out of the MAD window from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(LS_threshold is not None):
        if verbose: print("selecting genes based on Laplacian Score (LS) threshold: ",LS_threshold, "...")
        gene_selected = feature_selection.LS_selection(data_array, LS_threshold, 5, verbose)
        if verbose: print("removing", len(gene_selected) - sum(gene_selected), "genes under the LS threshold from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]


    ###########################################
    ######### Feature name Formating  #########
    ###########################################


    # Load data and perform dataset-specific processing
    if dataset_type == 'genomic':
        names = pd.DataFrame(names)
        # formatting gene names to get rid of the version number
        names['query'] = [n.split(".")[0] for n in names["Name"]]
        query_result = retrive_position(names, drop_na = True, verbose = verbose - 1)
        # Merge the original names DataFrame with the query results
        names = names.merge(query_result, on='query', how='left')
        names.reset_index(inplace=True)

    elif dataset_type == 'transcriptomic':
        names = pd.DataFrame([n.split("|") for n in names])
        names = names.set_axis([
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
        
    elif dataset_type == 'BRCA':
        names['query'] = names['gene_id'].apply(lambda x: x.split(".")[0])
        query_result = retrive_position(names, drop_na = False, verbose = verbose - 1)
        # Merge the original names DataFrame with the query results
        names = names.merge(query_result, on='query', how='left')
        names.reset_index(inplace=True)

    else:
        raise ValueError("Invalid dataset type. Choose 'genomic', 'transcriptomic' or 'BRCA'.")


    ###########################################
    ######## logical feature selection  #######
    ###########################################

    # logic based feature selection

    if(keep_only_symbols == True and dataset_type == 'genomic'):
        gene_selected = [False if s.startswith('ENSG') else True for s in names['name']]
        if verbose: print("removing", len(gene_selected) - sum(gene_selected), "not found symbols from the dataset, out of ", len(gene_selected))
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(drop_ambiguous_pos == True and dataset_type == 'genomic'):
        gene_selected = names["genomic_pos"].isna()
        if verbose: print("removing", len(gene_selected) - sum(gene_selected), "ambigously positioned symbols from the dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(keep_only_protein_coding and dataset_type == 'BRCA' ):
        gene_selected = names["gene_type"] == "protein_coding"
        if verbose: print("removing", len(gene_selected) - sum(gene_selected), "Non coding genes from dataset")
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    print("number of genes selected : ", len(data_array[0]))


    ###########################################
    ############## Sorting genes  #############
    ###########################################

    if(sort_symbols and dataset_type in ['genomic', 'BRCA'] ):
        if verbose: print("sorting based on genomic position chr then transcript start...")
        # reset the indexes because of all the previous transformations we have done
        names = names.reset_index(drop=True)
        names = names.sort_values(['genomic_pos.chr', 'genomic_pos.start'], ascending=[True, True])
        # Extract the sorted rows as a NumPy array
        data_array = data_array[:, names.index]


    ###########################################
    ############## normalisation  #############
    ###########################################
    
    if(normalization == True): 
        if verbose: print("normalizing data...")
        data_array = normalize(data_array)

    if(log1p == True): 
        if verbose: print("log(1 + x) transformation...")
        data_array = np.log1p(data_array)

    # after log1p transform because it already provide us with a very good dataset 
    if(min_max == True):
        if verbose: print("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data_array = scaler.fit_transform(data_array)

    if verbose: print("number of seq in the dataset :", len(data_array))


    metadata = {"name" : dataset_type,
                "feature_names" : names,
                "seq_names" : entries,
                "n_features" : len(data_array[0]),
                "n_seq" : len(entries),
                "meta_data" : meta_data,
                "subtypes" : subtypes} 
    
    return data_array, metadata
