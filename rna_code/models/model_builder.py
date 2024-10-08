"""Module to build model object

Returns
-------
Autoencoder
    Built auto encoder model

Raises
------
NotImplementedError
    If trying to build a model that isn't of type 'CNN' or 'MLP'.
"""

from rna_code.models.autoencoder import Autoencoder
from rna_code.models.cnn_ae import CNNAutoencoder
from rna_code.models.mlp_ae import MLPAutoencoder


class ModelBuilder:
    """Builder for model

    Parameters
    ----------
    shape : int
        Input size
    model_params : dict
        Parameters used for model building.
    """

    def __init__(self, shape: int, model_params: dict):
        self.shape = shape
        self.model_type = model_params.pop("model_type")
        self.model_params = model_params

        assert self.model_type in ["CNN", "MLP"], "model should be either CNN or MLP"

    def generate_model(self) -> Autoencoder:
        """Generate a model according to the model_params.

        Returns
        -------
        Autoencoder
            Built auto encoder

        Raises
        ------
        NotImplementedError
            If trying to build a model that isn't of type 'CNN' or 'MLP'.
        """
        match self.model_type:
            case "CNN":
                constructor = CNNAutoencoder
            case "MLP":
                constructor = MLPAutoencoder
            case _:
                raise NotImplementedError

        model = constructor(shape=self.shape, **self.model_params)
        model.build_encoder()
        model.build_decoder()
        return model
