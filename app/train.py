"""Whole training pipeline."""
import logging

from rna_code import CACHE_PATH
from rna_code.utils.experiment import Experiment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

logger.info("Done.")

config = {
    "dropout" : 0.5,
    "latent_dim" : 16,
    "model_type" : "CNN",
    "num_layers" :  3,
    "kernel_size" : 7,
    "padding" : 3,
    "n_epoch" : 1
}

config = {
    "dropout" : 0.3,
    "latent_dim" : 16,
    "model_type" : "MLP",
    "num_layers" :  8,
    "n_epoch" : 20,
    #"variational" : "VQ-VAE",
}

data_path  = CACHE_PATH / "data"

data_param = {"Path":data_path}

e = Experiment(data_param=data_param,  model_param=config)
e.run()
