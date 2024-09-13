import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, normalize

from .feature_selection.mad_selector import MADSelector
from .feature_selection.expression_selector import ExpressionSelector
from .feature_selection.laplacian_selector import LaplacianSelector
from .feature_selection.lasso_selector import LassoSelector

from .interface.BRCA_interface import BRCAInterface


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_dataset(
        dataset_type,
        MAD_threshold=None,
        LS_threshold=None,
        expression_threshold=None,
        normalization=False,
        keep_only_protein_coding = None,
        log1p=True,
        min_max=True,
        sort_symbols=False,
        ):
    
    """
    Generate a genomic or transcriptomic dataset from specified files, applying various
    processing and filtering steps.

    Args:
        dataset_type (str): Specifies the type of dataset to generate. Options are 'genomic' or 'transcriptomic'.
        subsample (int, optional): If specified, limits the number of samples to process.
        MAD_threshold (float, optional): Threshold for Median Absolute Deviation-based feature selection.
        LS_threshold (float, optional): Threshold for Laplacian Score-based feature selection.
        expression_threshold (float, optional): Threshold for expression level-based gene selection.
        normalization (bool, optional): If True, normalizes the data.
        log1p (bool, optional): If True, applies log(1 + x) transformation to the data.
        min_max (bool, optional): If True, scales the data using Min-Max normalization.
        sort_symbols (bool, optional): For genomic data, if True, sorts genes based on their genomic position.

    Returns:
        tuple: A tuple containing:
            - data_array (numpy.ndarray): The processed data array.
            - metadata (dict): A dictionary containing metadata information about the dataset.
    
    This function processes and prepares a dataset for analysis, including steps like
    loading data, metadata processing, feature selection, and normalization.

    """

    match dataset_type:
        case 'BRCA':
            data_interface = BRCAInterface()
        case _:
            raise NotImplementedError

    data_interface.setup()
    data_array = data_interface.dataset
    names = data_interface.names
    meta_data = data_interface.meta_data
    subtypes = data_interface.subtypes
    entry_names = data_interface.entry_names

    ###########################################
    ####### Numerical feature selection  ######
    ###########################################
    logger.debug("Numerical feature selection")


    # statistical feature selection
    expression_selector = ExpressionSelector(threshold = expression_threshold)
    mad_selector = MADSelector(threshold = MAD_threshold)
    laplacian_selector = LaplacianSelector(threshold= LS_threshold)

    if(expression_threshold is not None):
        gene_selected = expression_selector.select_features(data_array)
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(MAD_threshold is not None):
        gene_selected = mad_selector.select_features(data_array)
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    if(LS_threshold is not None):
        gene_selected = laplacian_selector.select_features(data_array)
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]


    ###########################################
    ######### Feature name Formatting  ########
    ###########################################
    logger.debug("Feature name formatting")

        
    if dataset_type == 'BRCA':
        names['query'] = names['gene_id'].apply(lambda x: x.split(".")[0])
        query_result = data_interface.retrive_position(names)
        # Merge the original names DataFrame with the query results
        names = names.merge(query_result, on='query', how='left')
        names.reset_index(inplace=True)


    ###########################################
    ######## logical feature selection  #######
    ###########################################
    logger.debug("logical feature selection ")

    # logic based feature selection

    if(keep_only_protein_coding and dataset_type == 'BRCA' ):
        gene_selected = names["gene_type"] == "protein_coding"
        logging.info("removing %i non coding genes from dataset", len(gene_selected) - sum(gene_selected))
        data_array = data_array[:,gene_selected]
        names = names[gene_selected]

    logger.debug("number of genes selected : %i", len(data_array[0]))


    ###########################################
    ############## Sorting genes  #############
    ###########################################

    if(sort_symbols and dataset_type in ['genomic', 'BRCA'] ):
        logging.info("sorting based on genomic position chr then transcript start...")
        # reset the indexes because of all the previous transformations we have done
        names = names.reset_index(drop=True)
        names = names.sort_values(['genomic_pos.chr', 'genomic_pos.start'], ascending=[True, True])
        # Extract the sorted rows as a NumPy array
        data_array = data_array[:, names.index]


    ###########################################
    ############## normalization  #############
    ###########################################

    if normalization:
        logging.info("normalizing data...")
        data_array = normalize(data_array)

    if log1p:
        logging.info("log(1 + x) transformation...")
        data_array = np.log1p(data_array)

    # after log1p transform because it already provide us with a very good dataset 
    if min_max:
        logging.info("scaling to [0, 1]...")
        scaler = MinMaxScaler(feature_range=(0, 1), clip = True)
        data_array = scaler.fit_transform(data_array)

    logging.info("number of seq in the dataset : %i", len(data_array))

    df = pd.DataFrame(
        data = data_array,
        index = entry_names,
        columns = names["gene_id"])

    meta_dict = {
        "meta_data" : meta_data,
        "subtypes" : subtypes,
    }
    meta_df = pd.DataFrame(
        meta_dict,
        index = entry_names)

    return df, meta_df
