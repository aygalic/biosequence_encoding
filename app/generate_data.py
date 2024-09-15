"""Generate and save data."""
import logging
import pickle
#import pandas as pd
import numpy as np

from rna_code import CACHE_PATH
from rna_code.data.dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info("Generating data..")

thresholds = {
    "LS_threshold" : 0.0020,
    "MAD_threshold" : 1,
}

builder = DatasetBuilder(dataset_type = "BRCA", selection_thresholds=thresholds)
df, meta_data = builder.generate_dataset()


logger.info("Saving data..")

data_path = CACHE_PATH / "data"/ 'BRCA_data.csv'
data_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(data_path)


#data_path = CACHE_PATH / "data"/ 'data_array.npy'
#data_path.parent.mkdir(parents=True, exist_ok=True)
#np.save(data_path, data_array)

metadata_path = CACHE_PATH / "data"/ 'meta_data.csv'
metadata_path.parent.mkdir(parents=True, exist_ok=True)
meta_data.to_csv(metadata_path)

#with metadata_path.open('wb') as f:
#    pickle.dump(meta_data, f)

logger.info("Done.")
