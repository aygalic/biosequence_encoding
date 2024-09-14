from abc import ABC
from typing import List, Optional, Any
import logging
import pandas as pd
import mygene
from pathlib import Path
class BaseInterface(ABC):

    def __init__(
            self,
            data_path : Path,
            metadata_path : Path):
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.subsample : int | None = None
        self.entries : list = []
        self.meta_data : Any
        self.names : Any

    @staticmethod
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
            logging.warning(f"File not found: {filename}")

    @staticmethod
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
            logging.warning(f"File not found: {filename}")
        except pd.errors.ParserError as e:
            logging.warning(f"Error parsing file {filename}: {e}")

    @staticmethod
    def retrive_position(names, drop_na=False):
        """
        Retrieve genomic positions for a list of gene names.

        Args:
            names (pd.DataFrame): DataFrame containing gene names.
            drop_na (bool, optional): Flag to drop NA values. Defaults to False.
            verbose (int, optional): Verbosity level.

        Returns:
            pd.DataFrame: DataFrame with retrieved genomic positions and symbols.
        """
        mg_client = mygene.MyGeneInfo()

        logging.debug("retriving %i symbols for genes",len(names))
        query_result = mg_client.querymany(names['query'], fields = ['genomic_pos', 'symbol'], scopes='ensembl.gene', species='human', verbose = False, as_dataframe = True)
        query_result = query_result.reset_index()
        logging.debug("Found %i symbols before duplicate removal", len(query_result))
        query_result = query_result.drop_duplicates(subset = ["query"])
        logging.debug(len(query_result), "symbols after duplicate removal")
        if drop_na:
            query_result['name'] = [q if(pd.isna(s)) else s for (s,q) in zip(query_result["symbol"],query_result["query"])]
        return query_result
