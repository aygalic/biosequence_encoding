"""
This module defines a collection of neural network models and modules,
primarily focusing on autoencoders, including variational autoencoders (VAE)
and vector quantized variational autoencoders (VQ-VAE), along with
additional attention mechanisms. The models are implemented using PyTorch.

Classes:
- `VectorQuantizer`: Implements the vector quantization mechanism used in VQ-VAE.
- `VectorQuantizerEMA`: Vector quantizer with Exponential Moving Average, used in VQ-VAE.
- `ResidualStack`: A residual stack module for building deep neural networks.
- `vq_conversion`: A linear conversion layer for VQ-VAE.
- `vq_pre_residual_stack_decoder`: Preprocessing layer before residual stacking in VQ-VAE decoder.
- `MultiHeadSelfAttention`: Implements multi-head self-attention using PyTorch's TransformerEncoder.
- `AttentionModule`: Basic attention module for feature weighting.
- `SelfAttention`: Custom self-attention mechanism for sequence data.
- `Autoencoder`: A flexible autoencoder class supporting various configurations including VAE,
VQ-VAE, convolutional layers, and attention mechanisms.

The module provides flexibility to construct various types of autoencoders with
different architectures and capabilities, suitable for a range of tasks from basic
auto-encoding to more complex generative modeling. Each class is equipped with
forward methods for defining the data flow through the network.

Example Usage:
    model = Autoencoder(shape=input_shape, latent_dim=64, variational='VAE')
    output = model(input_data)
"""
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import pytorch_lightning as pl

from rna_code.utils import helpers

from .residual_stack import ResidualStack
from .self_attention import SelfAttention
from .vector_quantizer import VectorQuantizer
from .vector_quantizer_EMA import VectorQuantizerEMA
from .vq_conversion import vq_conversion
from .vq_pre_residual_stack_decoder import vq_pre_residual_stack_decoder
from .attention_module import AttentionModule


class Autoencoder(pl.LightningModule):

    def __init__(
            self,
            shape,
            latent_dim=64,
            dropout = 0.1,
            slope = 0.05,
            num_layers = 3,
            variational = None,
            convolution = False,
            transformer = False,
            attention_size = 64,
            num_heads = 64,
            kernel_size = None,
            padding = None,
            num_embeddings = 512,
            embedding_dim = 512,
            decay = 0,
            commitment_cost = 1):
        
        super(Autoencoder, self).__init__()

        # Basic AE params
        self.input_shape = shape
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # CNN
        self.convolution = convolution
        self.kernel_size = kernel_size
        self.padding = padding

        # Attention
        self.transformer = transformer
        self.num_heads = num_heads
        self.attention_size = attention_size

        self.use_attention = False
        self.use_self_attention = False

        # VAE/VQ-VAE
        self.variational = variational
        self.num_embeddings = num_embeddings
        self.decay = decay

        if self.transformer:
            if self.num_heads is None:
                num_heads_candidate = helpers.find_primes(self.input_shape)
                if(len(num_heads_candidate) > 1):
                    self.num_heads = num_heads_candidate[-1]
                else:
                    self.num_heads = num_heads_candidate[-2]

            self.encoder_layers = TransformerEncoderLayer(d_model=self.input_shape, nhead=self.num_heads, dropout=self.dropout)
            self.encoder = nn.Sequential( 
                TransformerEncoder(self.encoder_layers, num_layers=self.num_layers),
                nn.LazyLinear(self.latent_dim)
            )

            decoder_layers = TransformerEncoderLayer(d_model=self.input_shape, nhead=self.num_heads, dropout=self.dropout)
            self.decoder = nn.Sequential(
                TransformerEncoder(decoder_layers, num_layers=self.num_layers),
                nn.Linear(self.input_shape, self.input_shape))  

        if convolution:
            if kernel_size is None:
                kernel_size = 7
                if padding is None:
                    padding = 3

            # Encoder
            encoder_layers = []
            in_channels = 1  # Starting with one input channel

            
            # Define the convolutional layers for the encoder and decoder
            for i in range(num_layers):
                out_channels = 32 * (2 ** i)
                encoder_layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.LeakyReLU(slope),
                    nn.Dropout(dropout),
                    nn.MaxPool1d(2)
                ])
                in_channels = out_channels

            encoder_layers.extend([
                nn.Flatten(),
                nn.LazyLinear(self.latent_dim),
                nn.LeakyReLU(slope)
            ])

            self.encoder = nn.Sequential(*encoder_layers)
            
            self.calculated_length = self._find_calculated_length()

            # Decoder
            decoder_layers = []
            in_features = self.latent_dim

            decoder_layers.extend([
                nn.Linear(in_features, in_channels * self.calculated_length),
                nn.Unflatten(1, (in_channels, self.calculated_length))
            ])

            #in_channels = 128
            for i in reversed(range(num_layers)):
                out_channels = 32 * (2 ** i)
                decoder_layers.extend([
                    nn.Upsample(scale_factor=2),
                    nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=padding),
                    nn.LeakyReLU(slope),
                    nn.Dropout(dropout)
                ])
                in_channels = out_channels

            # Last layer of decoder to reconstruct the input
            decoder_layers.extend([
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose1d(in_channels, 1, kernel_size=kernel_size, stride=2, padding=padding),
                nn.LazyLinear(self.input_shape),
                nn.Sigmoid()
            ])

            self.decoder = nn.Sequential(*decoder_layers)
        
        else:            
            # Encoder
            encoder_layers = []
            layer_sizes = [self.input_shape] + [1024 // (2 ** i) for i in range(num_layers - 1)] + [latent_dim]

            for i in range(num_layers):
                encoder_layers.extend([
                    nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                    nn.LeakyReLU(slope),
                    nn.Dropout(dropout)
                ])

            self.encoder = nn.Sequential(*encoder_layers)

            # Decoder
            decoder_layers = []
            layer_sizes.reverse()

            for i in range(num_layers):
                decoder_layers.extend([
                    nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                    nn.LeakyReLU(slope),
                    nn.Dropout(dropout)
                ])

            decoder_layers.append(nn.Sigmoid())

            self.decoder = nn.Sequential(*decoder_layers)


        if self.variational == "VAE":
            # For VAE, create additional layers to learn log_var
            self.mu_layer = nn.Linear(self.latent_dim, self.latent_dim)
            self.logvar_layer = nn.Linear(self.latent_dim, self.latent_dim)
        
        if self.variational == "VQ-VAE":
            # for VQ-VAE, we have a series of new element that need to be added to support the quantization
            self.encoder_residual_stack = ResidualStack(self.latent_dim)
            self.pre_vq_conv = vq_conversion(self.latent_dim, num_embeddings)
            if self.decay > 0.0:
                self.vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, self.decay)
            else:
                self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
            self.pre_vq_decoder = vq_pre_residual_stack_decoder(self.num_embeddings, self.latent_dim, self.dropout)
            self.decoder_residual_stack = ResidualStack(self.latent_dim)

    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, list) else batch
        # Forward pass
        if self.variational == "VAE":
            x_reconstructed, mu, log_var = self(x)
            # Calculate VAE loss
            recon_loss = F.mse_loss(x_reconstructed, x)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_loss
        elif self.variational == "VQ-VAE":
            vq_loss, x_reconstructed, perplexity, _, _ = self(x)
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

    
    def _find_calculated_length(self):
            mock_input = torch.rand(1, 1, self.input_shape)  
            with torch.no_grad():
                self.eval()
                for layer in self.encoder:
                    if isinstance(layer, nn.Flatten):
                        break 
                    mock_input = layer(mock_input)
            calculated_length = mock_input.size()
            return calculated_length[2]
    
    def add_attention(self):
        """Integrate the attention module after the encoder and before the decoder."""
        self.use_attention = True
        # Attention for the convolutional path

        self.attention_module = AttentionModule(in_features = self.latent_dim)  
        
    def add_self_attention(self, attention_dropout = 0.2):
        self.use_self_attention = True
        self.attention_module = SelfAttention(self.input_shape, self.attention_size, attention_dropout)


    def encode(self, x):
        x = self.encoder(x)

        if self.use_attention:
            x = self.attention_module(x)
        
        if self.variational == "VAE":
            mean, logvar = self.mu_layer(x), self.logvar_layer(x)
            return mean, logvar
        
        elif self.variational == "VQ-VAE":
            x = self.encoder_residual_stack(x)

        else:
            return x
        


    def decode(self, x):
        # VQ-VAE have to pre-decode the quantized space since it isn't the same dimension as the latent space 
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
        #breakpoint()

        #x = x[0]
        # Apply attention here if using convolution
        if self.use_self_attention:
            x = self.attention_module(x)


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


