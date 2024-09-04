"""Generate and save data."""
import logging
import pickle

import numpy as np

from rna_code import CACHE_PATH
from rna_code.data import data_handler

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info("Generating data..")

data_array, meta_data = data_handler.generate_dataset(
            dataset_type = "BRCA",
            LS_threshold= 0.0020,
            MAD_threshold = 1, 
        )


logger.info("Saving data..")

data_path = CACHE_PATH / "data"/ 'data_array.npy'
data_path.parent.mkdir(parents=True, exist_ok=True)
np.save(data_path, data_array)

metadata_path = CACHE_PATH / "data"/ 'meta_data.json'
metadata_path.parent.mkdir(parents=True, exist_ok=True)
with metadata_path.open('wb') as f:
    pickle.dump(meta_data, f)

logger.info("Done.")
