import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feature_size, num_heads, dropout):
        super(MultiHeadSelfAttention, self).__init__()

        self.feature_size = feature_size  # The input size (dimension) of each time step of each sequence
        self.num_heads = num_heads  # Number of attention heads
        self.dropout = dropout  # Dropout rate

        # Define the encoder layer using PyTorch's TransformerEncoderLayer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=self.feature_size,  # The feature size
            nhead=self.num_heads,  # Number of heads in the multiheadattention models
            dropout=self.dropout  # Dropout rate
        )

        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=1)  # We use one layer here for simplicity

    def forward(self, x):
        """
        Forward pass of the multi-head attention layer.

        Arguments:
        x -- A tensor of shape (batch_size, sequence_length, feature_size)

        Returns:
        out -- Multi-head self-attention applied output
        """
        # Transformer expects inputs of shape (sequence_length, batch_size, feature_size)
        x = x.permute(1, 0, 2)

        # Apply the transformer encoder
        out = self.transformer_encoder(x)

        # Return output to the original shape (batch_size, sequence_length, feature_size)
        out = out.permute(1, 0, 2)
        
        return out




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
    
class SelfAttention(nn.Module):
    def __init__(self, sequence_length, attention_size, attention_dropout = 0.2):
        super(SelfAttention, self).__init__()
        self.sequence_length = sequence_length
        self.attention_size = attention_size
        self.attention_dropout = attention_dropout
        
        self.query_layer = nn.Linear(sequence_length, attention_size, bias=False)
        self.key_layer = nn.Linear(sequence_length, attention_size, bias=False)
        self.value_layer = nn.Linear(sequence_length, sequence_length, bias=False)

    def forward(self, x):
        # x shape: (batch_size, 1, sequence_length)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)
        
        # Calculate attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.attention_size)
        attention = F.softmax(scores, dim=-1)
        attention = F.dropout(attention, self.attention_dropout)
        # Multiply scores with value layer
        attended_values = torch.matmul(attention, value)
        return attended_values
    



class Autoencoder(nn.Module):
    def find_calculated_length(self):
            # Create a mock tensor. Assuming it's a 1D signal, the "1" size is for the channel dimension.
            # The batch size here is 1, and it could be any number since it doesn't affect the calculation.
            mock_input = torch.rand(1, 1, self.input_shape)  # Adapt the shape according to your specific use case

            # Pass the mock input through the encoder layers before flattening.
            with torch.no_grad():  # We do not need gradients for this operation
                self.eval()  # Set the model to evaluation mode
                for layer in self.encoder:
                    if isinstance(layer, nn.Flatten):
                        break  # Stop right before the Flatten layer
                    mock_input = layer(mock_input)  # Pass the mock input through

            # The resulting mock_input contains the size after all transformations
            calculated_length = mock_input.size()  # Extract the length (size of the spatial dimension)

            return calculated_length[2]
    

    def __init__(
            self, 
            shape, 
            latent_dim=64, 
            dropout = 0.1, 
            slope = 0.05, 
            num_layers = 16,
            variational=False, 
            convolution = False, 
            transformer = False,
            attention_size = 64,
            num_heads = 64,
            kernel_size = None,
            padding = None):
        
        super(Autoencoder, self).__init__()
        self.input_shape = shape
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.variational = variational
        self.convolution = convolution
        self.use_attention = False
        self.use_self_attention = False
        self.transformer = transformer
        self.attention_size = attention_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.dropout = dropout
        self.num_heads = num_heads
        
        if self.transformer: 

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
            
            # Define the convolutional layers for the encoder and decoder
            self.encoder = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size = kernel_size, stride=2, padding = padding),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),  # Pooling layer
                
                nn.Conv1d(32, 64, kernel_size = kernel_size, stride=2, padding = padding),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),  # Pooling layer
                
                nn.Conv1d(64, 128, kernel_size = kernel_size, stride=2, padding = padding),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),  # Pooling layer
                
                nn.Flatten(),
                nn.LazyLinear(latent_dim),  # Adjust based on strides, kernels, and pooling
                nn.LeakyReLU(slope)
            )
            
            self.calculated_length = self.find_calculated_length()

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128 * (self.calculated_length)),  # Adjusted due to pooling layers
                nn.Unflatten(1, (128, self.calculated_length)),
                
                nn.Upsample(scale_factor=2),  # Upsampling layer
                nn.ConvTranspose1d(128, 64, kernel_size = kernel_size, stride=2, padding=padding),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Upsample(scale_factor=2),  # Upsampling layer
                nn.ConvTranspose1d(64, 32, kernel_size = kernel_size, stride=2, padding=padding),
                nn.LeakyReLU(slope),
                nn.Dropout(dropout),
                
                nn.Upsample(scale_factor=2),  # Upsampling layer
                nn.ConvTranspose1d(32, 1, kernel_size = kernel_size, stride=2, padding=padding),  # Adjusted kernel size
                #nn.LeakyReLU(slope),  # Preserved the non-linearity
                
                nn.LazyLinear(self.input_shape), 
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

        if variational:
                # For VAE, create additional layers to learn log_var
                self.mu_layer = nn.Linear(latent_dim, latent_dim)
                self.logvar_layer = nn.Linear(latent_dim, latent_dim)
    
    def add_attention(self):
        """Integrate the attention module after the encoder and before the decoder."""
        self.use_attention = True
        # Attention for the convolutional path

        self.attention_module = AttentionModule(in_features = self.latent_dim)  
        
    def add_self_attention(self, attention_dropout = 0.2):
        self.use_self_attention = True
        self.attention_module = SelfAttention(self.input_shape, self.attention_size, attention_dropout)


    def encode(self, x):
        if self.transformer: 
            #x = x.permute(1, 0, 2)
            None

        x = self.encoder(x)

        if self.use_attention:
            x = self.attention_module(x)
        

        if self.variational:
            mean, logvar = self.mu_layer(x), self.logvar_layer(x)
            return mean, logvar
        else:
            return x
        
    def decode(self, x):
        x = self.decoder(x)
        if self.transformer: 
            #x = x.permute(1, 0, 2)
            None
        return x

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, x):
        # Apply attention here if using convolution
        if self.use_self_attention:
            x = self.attention_module(x)


        if self.variational:
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            x_reconstructed = self.decoder(z)
            return x_reconstructed, mu, log_var


        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed
    



