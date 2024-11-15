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

logger.info("Merging data..")

def main():
    data_path_BRCA = CACHE_PATH / "data"/ 'BRCA_data.csv'
    data_path_CPTAC_3 = CACHE_PATH / "data"/ 'CPTAC_3_data.csv'
    data_BRCA = pd.read_csv(data_path_BRCA, index_col=0).values
    data_CPTAC_3 = pd.read_csv(data_path_CPTAC_3, index_col=0).values
    merged_data = DatasetMerger.intersect(data_BRCA, data_CPTAC_3)
    


if __name__ == "__main__":
    main()