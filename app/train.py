"""Whole training pipeline."""

import logging

from rna_code import CACHE_PATH
from rna_code.utils.experiment import Experiment

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logging.getLogger("fsspec.local").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

config = {
    "dropout": 0.3,
    "latent_dim": 16,
    "model_type": "MLP",
    "num_layers": 8,
    "n_epoch": 20,
}

data_path = CACHE_PATH / "data"

data_param = {"Path": data_path}


def main():
    """Handle the training of the model using a simple straightforward approach."""
    logger.info("Training model.")
    e = Experiment(data_param=data_param, model_param=config)
    e.run()
    logger.info("Done.")


if __name__ == "__main__":
    main()
