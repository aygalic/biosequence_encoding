from typing import Literal

from rna_code.models.mlp_ae import MLPAutoencoder
from rna_code.models.cnn_ae import CNNAutoencoder


class ModelBuilder:
    def __init__(
            self,
            shape,
            model_params: dict):
        self.shape = shape

        self.model_type = model_params.pop("model_type")
        assert self.model_type in ["CNN", "MLP"]

        self.variational = model_params.get("variational", "False")
        assert self.variational in ["False", "VAE", "VQ-VAE"]

        self.model_params = model_params

    def generate_model(self):
        match self.model_type:
            case "CNN":
                constructor = CNNAutoencoder
            case "MLP":
                constructor = MLPAutoencoder
            case _:
                raise NotImplementedError
            
        model = constructor(self.shape, **self.model_params)
        model.build_encoder()
        model.build_decoder()
        return model
