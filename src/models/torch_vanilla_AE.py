import os
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, shape, dropout, latent_dim):
        super(Autoencoder, self).__init__()

        
        # Encoder
        self._encoder = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shape, 1024),
            nn.LeakyReLU(0.05),
            
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.05),
            
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.05),
            
            nn.Linear(256, latent_dim),
            nn.LeakyReLU(0.05)
        )
        
        # Decoder
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.5),
            
            nn.Linear(1024, shape),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self._encoder(x)
        x = self._decoder(x)
        return x


