"""Interface with the BRCA dataset."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .. import BRCA_DATA_PATH, BRCA_METADATA_FILE, BRCA_SUBTYPES_FILE
from .base_interface import BaseInterface


class BRCAInterface(BaseInterface):
    """Interface wth app and file system for the BRCA dataset.

    Parameters
    ----------
    data_path : Path, optional
        Path of the directory containing the data, by default BRCA_DATA_PATH
    metadata_path : Path, optional
        Path of the metadata file, by default BRCA_METADATA_FILE
    """

    def __init__(
        self, data_path: Path = BRCA_DATA_PATH, metadata_path: Path = BRCA_METADATA_FILE
    ):
        super().__init__(data_path, metadata_path)
        self.subtypes_table: Path = BRCA_SUBTYPES_FILE  # provided by supervisor
        self.subtypes: list = []
        self.data_array: np.ndarray

    def _prepare_entires(self):
        """Select relevant entries from file system."""
        all_files = list(self.data_path.rglob("*"))
        self.entries = [
            file
            for file in all_files
            if str(file).endswith("augmented_star_gene_counts.tsv")
        ]
        self.entries.sort()

    def _load_metadata(self):
        """Load metadata based on filepath."""
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.meta_data = json.load(f)

    def find_subtypes(self):
        """Find subtypes associated with each observation based on subtype file."""
        # Step 1: Construct a mapping from file_name to entity_submitter_id
        file_to_id = {}
        for item in self.meta_data:
            file_name = item["file_name"][:-4]  # remove the .tsv extension
            entity_id = item["associated_entities"][0]["entity_submitter_id"]
            file_to_id[file_name] = entity_id
        # Step 2: For each file name in metadata["sequence_names"], find its corresponding entity_submitter_id
        patient_id = []
        for file_name in self.entries:
            if file_name.stem in file_to_id.keys():
                patient_id.append(file_to_id[file_name.stem])
        subtypes_table = pd.read_csv(self.subtypes_table, index_col=0)
        subtypes_dict = {
            str(index)[:12]: subtype for index, subtype in subtypes_table.itertuples()
        }
        self.subtypes = [
            subtypes_dict.get(identifier[:12], "Unknown") for identifier in patient_id
        ]

    def load_patients(self):
        """Load patients based on pre computed entries"""
        logging.info("loading samples...")
        self.data_array = np.array(
            [self.load_patient_data(e, header=5) for e in self.entries]
        )
        logging.info("loaded %i samples.", len(self.data_array))

    def _retrieve_gene_position(self):
        """Use mygene to retrieve gene position"""
        self.names["query"] = self.names["gene_id"].apply(lambda x: x.split(".")[0])
        query_result = self.retrieve_position(self.names)
        self.names = self.names.merge(query_result, on="query", how="left")

    def setup(self):
        """Perform all necessary steps to provide with a dataset."""
        self._load_metadata()
        self._prepare_entires()
        if self.subsample is not None:
            self.entries = self.entries[: self.subsample]
        self.load_patients()
        self.names = pd.DataFrame(
            self.get_gene_names_from_file(
                self.entries[0], header=1, skiprows=[2, 3, 4, 5]
            )
        )
        self.find_subtypes()
        self._retrieve_gene_position()

    @property
    def entry_names(self) -> list[str]:
        """Get entries names

        Returns
        -------
        list[str]
            List containing the name for each observation
        """

        return [str(e).split("/")[9] for e in self.entries]
