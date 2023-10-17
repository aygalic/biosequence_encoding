import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, in_features):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Tanh(),
            nn.Linear(in_features, 1),
            nn.Softmax(dim=1)  # Softmax over the sequence dimension.
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return attention_weights * x
    
class Autoencoder(nn.Module):
    def __init__(self, shape, latent_dim=64, dropout = 0.1, slope = 0.05, is_variational=False, use_convolution = False):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.is_variational = is_variational
        self.use_convolution = use_convolution
        self.use_attention = False

        if use_convolution:
            # Define the convolutional layers for the encoder and decoder
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),  # Pooling layer
                
                nn.Conv1d(32, 64, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),  # Pooling layer
                
                nn.Conv1d(64, 128, kernel_size=3, stride=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),  # Pooling layer
                
                nn.Flatten(),
                nn.LazyLinear(latent_dim),  # Adjust based on strides, kernels, and pooling
                nn.LeakyReLU(slope)
            )

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128 * (shape // 64)),  # Adjusted due to pooling layers
                nn.Unflatten(1, (128, shape // 64)),
                
                nn.Upsample(scale_factor=2),  # Upsampling layer
                nn.ConvTranspose1d(128, 64, kernel_size=7, stride=2, padding=2),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Upsample(scale_factor=2),  # Upsampling layer
                nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=0),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Upsample(scale_factor=2),  # Upsampling layer
                nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=2),  # Adjusted kernel size
                #nn.LeakyReLU(slope),  # Preserved the non-linearity

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
    
    def add_attention(self):
        """Integrate the attention module after the encoder and before the decoder."""
        self.use_attention = True
        # Attention for the convolutional path

        self.conv_attention = AttentionModule(in_features = self.latent_dim)  
        

    def encode(self, x):
        x = self.encoder(x)

        # Apply attention here if using convolution
        if self.use_attention:
            x = self.conv_attention(x)

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
    



