"""Whole training pipeline."""
import logging
import pickle

import numpy as np

from rna_code import CACHE_PATH
from rna_code.utils.experiment import Experiment


logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


logger.info("Loading data..")

data_path = CACHE_PATH / "data"/ 'data_array.npy'
data_path.parent.mkdir(parents=True, exist_ok=True)
data_array = np.load(data_path)

metadata_path = CACHE_PATH / "data"/ 'meta_data.json'
metadata_path.parent.mkdir(parents=True, exist_ok=True)
with metadata_path.open('rb') as f:
    meta_data = pickle.load(f)

logger.info("Done.")

#data_param = '../workfiles/light_BRCA_ds.pkl'
config = {
    "dropout" : 0.5,
    "latent_dim" : 16,
    "convolution" : True,
    "num_layers" :  3,
    "kernel_size" : 7,
    "padding" : 3,
    "n_epoch" : 12
}


data_param = {
    "dataset_type" : "BRCA",
    "LS_threshold": 0.0020,
    "MAD_threshold" : 1, 
}


e = Experiment(data_param=data_param, model_param=config)
e.run()