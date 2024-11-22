"""Whole training pipeline."""

import logging

from rna_code import CACHE_PATH
from rna_code.utils.transfert_leanring_experiment import TransfertLearningExperiment

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
    # "variational" : "VQ-VAE",
}

data_path = CACHE_PATH / "data_transfert_learning"

data_param = {"Path": data_path}


def main():
    """Handle the training of the model using a transfer learning approach."""
    logger.info("Training model using transfer learning.")
    e = TransfertLearningExperiment(data_param=data_param, model_param=config)
    e.run()
    logger.info("Done.")


if __name__ == "__main__":
    main()
