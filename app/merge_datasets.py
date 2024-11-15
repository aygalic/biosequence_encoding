"""Merge datasets."""
import logging
import pandas as pd 


from rna_code import CACHE_PATH
from rna_code.utils.dataset_merger import DatasetMerger

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)


def main():
    logger.info("Merging data..")
    data_path_BRCA = CACHE_PATH / "data"/ 'BRCA_data.csv'
    data_path_CPTAC_3 = CACHE_PATH / "data"/ 'CPTAC_3_data.csv'
    data_BRCA = pd.read_csv(data_path_BRCA, index_col=0)
    data_CPTAC_3 = pd.read_csv(data_path_CPTAC_3, index_col=0)
    merged_data = DatasetMerger.intersect(data_BRCA, data_CPTAC_3)
    data_BRCA = merged_data.loc[data_BRCA.index]
    data_CPTAC_3 = merged_data.loc[data_CPTAC_3.index]
    data_BRCA.to_csv(CACHE_PATH / "data"/ 'BRCA_data_intersect.csv')
    data_CPTAC_3.to_csv(CACHE_PATH / "data"/ 'CPTAC_3_data_intersect.csv')

if __name__ == "__main__":
    main()