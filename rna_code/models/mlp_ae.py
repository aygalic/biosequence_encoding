import torch.nn as nn
from .autoencoder import Autoencoder

class MLPAutoencoder(Autoencoder):

    def __init__(
            self,
            shape,
            latent_dim=64,
            dropout = 0.1,
            slope = 0.05,
            num_layers = 3,
            variational = None,
            num_embeddings = 512,
            embedding_dim = 512,
            commitment_cost = 1):
        
        super().__init__(
            shape = shape,
            latent_dim=latent_dim,
            dropout = dropout,
            slope = slope,
            num_layers = num_layers,
            variational = variational,
            num_embeddings = num_embeddings,
            embedding_dim = embedding_dim,
            commitment_cost = commitment_cost)
        
        self.layer_sizes = [self.input_shape] + [1024 // (2 ** i) for i in range(self.num_layers - 1)] + [latent_dim]

    def build_encoder(self):
        encoder_layers = []
        for i in range(self.num_layers):
            encoder_layers.extend([
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]),
                nn.LeakyReLU(self.slope),
                nn.Dropout(self.dropout)
            ])
        self.encoder = nn.Sequential(*encoder_layers)

    def build_decoder(self):
        """build decoder"""
        decoder_layers = []
        self.layer_sizes.reverse()
        for i in range(self.num_layers):
            decoder_layers.extend([
                nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]),
                nn.LeakyReLU(self.slope),
                nn.Dropout(self.dropout)
            ])
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)
