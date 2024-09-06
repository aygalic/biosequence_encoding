"""Whole training pipeline."""
import logging

from rna_code import CACHE_PATH
from rna_code.utils.experiment import Experiment

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Done.")

config = {
    "dropout" : 0.5,
    "latent_dim" : 16,
    "convolution" : True,
    "num_layers" :  3,
    "kernel_size" : 7,
    "padding" : 3,
    "n_epoch" : 12
}

data_path  = CACHE_PATH / "data"

e = Experiment(data_param=data_path,  model_param=config)
e.run()
