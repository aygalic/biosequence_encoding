import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, shape, latent_dim=64, dropout = 0.1, is_variational=False):
        super(Autoencoder, self).__init__()
        self.is_variational = is_variational
        
        # Encoder
        self._encoder = nn.Sequential(
            nn.Linear(shape, 1024),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            
            nn.Linear(256, latent_dim),
            nn.LeakyReLU(0.05)
        )
        
        if is_variational:
            # For VAE, create additional layers to learn log_var
            self.mu_layer = nn.Linear(256, latent_dim)
            self.logvar_layer = nn.Linear(256, latent_dim)
        
        # Decoder
        self._decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.05),
            nn.Dropout(dropout),
            
            nn.Linear(1024, shape),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        if self.is_variational:
            x = self._encoder(x)
            mu = self.mu_layer(x)
            log_var = self.logvar_layer(x)
            z = self.reparameterize(mu, log_var)
        else:
            z = self._encoder(x)
            
        x_reconstructed = self._decoder(z)
        return x_reconstructed
    
    # Optionally: If you wish to return mu and log_var during training for loss calculation
    # (e.g., for the Kullback-Leibler divergence term), you might adjust the forward function 
    # and training loop accordingly.


