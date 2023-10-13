import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, shape, latent_dim=64, dropout = 0.1, slope = 0.05, is_variational=False, use_convolution = False):
        super(Autoencoder, self).__init__()
        self.is_variational = is_variational
        self.use_convolution = use_convolution

        if use_convolution:
            # Define the convolutional layers for the encoder and decoder
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Conv1d(32, 64, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Conv1d(64, 128, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Flatten(),
                nn.LazyLinear(latent_dim),  # Adjust based on strides and kernels
                nn.LeakyReLU(slope)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128 * (shape // 8)),
                nn.Unflatten(1, (128, shape // 8)),
                
                nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.ConvTranspose1d(32, 1, kernel_size=1, stride=2),
                nn.Sigmoid()
            )
        else:
            # Encoder
            
            self.encoder = nn.Sequential(
                nn.Linear(shape, 1024),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Linear(1024, 512),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Linear(512, 256),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Linear(256, latent_dim),
                nn.LeakyReLU(slope)
            )

            # Decoder        
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 256),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Linear(256, 512),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Linear(512, 1024),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Linear(1024, shape),
                nn.Sigmoid()
            )

        if is_variational:
                # For VAE, create additional layers to learn log_var
                self.mu_layer = nn.Linear(latent_dim, latent_dim)
                self.logvar_layer = nn.Linear(latent_dim, latent_dim)
    
    def encode(self, x):
        x = self.encoder(x)
        if self.is_variational:
            mean, logvar = self.mu_layer(x), self.logvar_layer(x)
            return mean, logvar
        else:
            return x
        
    def decode(self, x):
        return self.decoder(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        
        if self.is_variational:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_reconstructed = self.decoder(z)
            return x_reconstructed, mu, log_var


        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
    



