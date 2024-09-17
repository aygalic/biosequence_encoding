from abc import ABC, abstractmethod
import torch

import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl

from .residual_stack import ResidualStack
from .vector_quantizer import VectorQuantizer
from .vq_conversion import vq_conversion
from .vq_pre_residual_stack_decoder import vq_pre_residual_stack_decoder

class Autoencoder(pl.LightningModule, ABC):

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

        self.encoder : nn.Module
        self.decoder : nn.Module

        if self.variational == "VAE":
            # For VAE, create additional layers to learn log_var
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.logvar_layer = nn.Linear(self.latent_dim, self.latent_dim)
        
        if self.variational == "VQ-VAE":
            # for VQ-VAE, we have a series of new element that need to be added to support the quantization
            self.encoder_residual_stack = ResidualStack(self.latent_dim)
            self.pre_vq_conv = vq_conversion(self.latent_dim, num_embeddings)
            self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            self.pre_vq_decoder = vq_pre_residual_stack_decoder(self.num_embeddings, self.latent_dim, self.dropout)
            self.decoder_residual_stack = ResidualStack(self.latent_dim)

    @abstractmethod
    def build_encoder(self):
        """build encoder"""
        
    @abstractmethod
    def build_decoder(self):
        """build decoder"""

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        if self.variational == "VAE":
            x_reconstructed, mu, log_var = self(x)
            # Calculate VAE loss
            recon_loss = F.mse_loss(x_reconstructed, x)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss
        elif self.variational == "VQ-VAE":
            vq_loss, x_reconstructed, _, _, _ = self(x)
            recon_loss = F.mse_loss(x_reconstructed, x)
            loss = recon_loss + vq_loss
        else:
            x_reconstructed = self(x)
            loss = F.mse_loss(x_reconstructed, x)

        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def encode(self, x):
        x = self.encoder(x)
        if self.variational == "VAE":
            mean, logvar = self.mu_layer(x), self.logvar_layer(x)
            return mean, logvar
        elif self.variational == "VQ-VAE":
            x = self.encoder_residual_stack(x)
        return x

    def decode(self, x):
        # VQ-VAE have to pre-decode the quantized space since it isn't the same
        # dimension as the latent space 
        if self.variational == "VQ-VAE":
            x = self.pre_vq_decoder(x)
            x = self.decoder_residual_stack(x)

        x = self.decoder(x)
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        if self.variational == "VAE":
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_reconstructed = self.decoder(z)
            return x_reconstructed, mu, log_var
        
        elif self.variational == "VQ-VAE":
            z = self.encoder(x)
            z = self.pre_vq_conv(z)
            loss, quantized, perplexity, encodings = self.vq_vae(z)
            x_recon = self.decode(quantized)
            return loss, x_recon, perplexity, encodings, quantized

        else:
            z = self.encode(x)
            x_reconstructed = self.decode(z)
            return x_reconstructed


