"""Base module for auto encoders."""

from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual_stack import ResidualStack
from .vector_quantizer import VectorQuantizer
from .vq_conversion import vq_conversion
from .vq_pre_residual_stack_decoder import vq_pre_residual_stack_decoder


class Autoencoder(pl.LightningModule, ABC):
    """Base module for auto encoder.

    Parameters
    ----------
    shape : int
        Number of input variable
    latent_dim : int, optional
        Size of the latent vector, by default 64
    dropout : float, optional
        dropout rate, by default 0.1
    slope : float, optional
        Leaky ReLU slope, by default 0.05
    num_layers : int, optional
        number of encoder/decoder layers, by default 3
    variational : bool | str, optional
        Whether or not to use variational autoencoder, can be VAE for variational
        autoencoder, VQ-VAE for vector quantized variational autoencoder or None if
        no variational logic, by default None
    num_embeddings : int, optional
        VQ only, number of embeddings, by default 512
    embedding_dim : int, optional
        VQ only, size of the embedding dimension for the encoding vectors, by
        default 512
    commitment_cost : int, optional
        VQ only, commitment cost, by default 1
    """

    def __init__(
        self,
        shape: int,
        latent_dim: int = 64,
        dropout: float = 0.1,
        slope: float = 0.05,
        num_layers: int = 3,
        variational: bool | str = None,
        num_embeddings: int = 512,
        embedding_dim: int = 512,
        commitment_cost: int = 1,
    ):
        super(Autoencoder, self).__init__()

        # Basic AE params
        self.input_shape = shape
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.slope = slope

        # VAE/VQ-VAE
        self.variational = variational
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.encoder: nn.Module
        self.decoder: nn.Module

        if self.variational == "VAE":
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.mu: torch.Tensor
            self.logvar_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.log_var: torch.Tensor

        elif self.variational == "VQ-VAE":
            self.vq_loss: torch.Tensor
            self.perplexity: torch.Tensor
            self.encoder_residual_stack = ResidualStack(self.latent_dim)
            self.pre_vq_conv = vq_conversion(self.latent_dim, self.embedding_dim)
            self.quantizer = VectorQuantizer(
                num_embeddings, embedding_dim, commitment_cost
            )
            self.pre_vq_decoder = vq_pre_residual_stack_decoder(
                self.embedding_dim, self.latent_dim, self.dropout
            )
            self.decoder_residual_stack = ResidualStack(self.latent_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """encoding logic to produce latent vector

        Parameters
        ----------
        x : torch.Tensor
            Input vector

        Returns
        -------
        torch.Tensor
            Latent vector
        """
        x = self.encoder(x)
        if self.variational == "VAE":
            mu = self.mu_layer(x)
            return mu
        if self.variational == "VQ-VAE":
            x = self.encoder_residual_stack(x)
            x = self.pre_vq_conv(x)
            _, quantized, _, encodings = self.quantizer(x)
            return quantized
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function that reconstruct the input through the latent space

        Parameters
        ----------
        x : torch.Tensor
            Input vector

        Returns
        -------
        torch.Tensor
            Reconstruction of the input vector
        """
        x = self.encoder(x)
        if self.variational == "VAE":
            self.mu, self.log_var = self.mu_layer(x), self.logvar_layer(x)
            x = self.reparameterize(self.mu, self.log_var)
        elif self.variational == "VQ-VAE":
            x = self.encoder_residual_stack(x)
            x = self.pre_vq_conv(x)
            self.vq_loss, quantized, self.perplexity, encodings = self.quantizer(x)
            x = self.pre_vq_decoder(quantized)
            x = self.decoder_residual_stack(x)
        x_recon = self.decoder(x)
        return x_recon

    def training_step(self, batch, batch_idx):
        # x = batch[0] if isinstance(batch, list) else batch
        x = batch[0]
        x_reconstructed = self(x)
        loss = F.mse_loss(x_reconstructed, x)
        if self.variational == "VAE":
            kl_loss = -0.5 * torch.sum(
                1 + self.log_var - self.mu.pow(2) - self.log_var.exp()
            )
            loss += kl_loss
        elif self.variational == "VQ-VAE":
            loss += +self.vq_loss

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparametrization trick for VAE

        Parameters
        ----------
        mu : torch.Tensor
            Mean tensor for VAE sampling
        log_var : torch.Tensor
            Log variance for VAE sampling

        Returns
        -------
        torch.Tensor
            Z sampled variable.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @abstractmethod
    def build_encoder(self):
        """build encoder"""

    @abstractmethod
    def build_decoder(self):
        """build decoder"""
