"""Merge datasets."""

import logging
import shutil

import pandas as pd

from rna_code import CACHE_PATH
from rna_code.utils.dataset_merger import DatasetMerger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Take 2 existing dataset (BRCA and CPTAC-3) and create 2 new datasets in order to
    match feature number/order for consistency during transfer learning."""
    logger.info("Merging data...")
    data_path_BRCA = CACHE_PATH / "data" / "BRCA_data.csv"
    data_path_CPTAC_3 = CACHE_PATH / "data" / "CPTAC_3_data.csv"
    data_BRCA = pd.read_csv(data_path_BRCA, index_col=0)
    data_CPTAC_3 = pd.read_csv(data_path_CPTAC_3, index_col=0)
    merged_data = DatasetMerger.intersect(data_BRCA, data_CPTAC_3)
    data_BRCA = merged_data.loc[data_BRCA.index]
    data_CPTAC_3 = merged_data.loc[data_CPTAC_3.index]

    merged_data_path = CACHE_PATH / "data_transfert_learning"
    merged_data_path.mkdir(parents=True, exist_ok=True)
    data_BRCA.to_csv(merged_data_path / "BRCA_data.csv")
    data_CPTAC_3.to_csv(merged_data_path / "CPTAC_3_data.csv")

    logger.info("Copying metadata...")
    files_to_copy = [
        CACHE_PATH / "data" / "meta_data.csv",
        CACHE_PATH / "data" / "CPTAC_3_meta_data.csv",
    ]
    destination_directory = merged_data_path
    _ = [
        shutil.copy(file_to_copy, destination_directory)
        for file_to_copy in files_to_copy
    ]

    logger.info("Done.")


if __name__ == "__main__":
    main()
