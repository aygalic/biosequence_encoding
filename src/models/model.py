import torch
import torch.nn as nn
import torch.nn.functional as F
import math 
from einops import rearrange

from torch.nn import TransformerEncoder, TransformerEncoderLayer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        #inputs = rearrange(inputs, 'b c l -> b l c')
        inputs = inputs.contiguous()
        #inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        #quantized = rearrange(quantized, 'b c l -> b l c')
        quantized = quantized.contiguous()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        #return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, quantized, perplexity, encodings
    
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        #inputs = rearrange(inputs, 'b c l -> b l c')
        inputs = inputs.contiguous()
        #inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        #quantized = rearrange(quantized, 'b c l -> b l c')
        quantized = quantized.contiguous()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
               
        # convert quantized from BHWC -> BCHW
        #return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
        return loss, quantized, perplexity, encodings   
    
class ResidualStack(nn.Module):
    def __init__(self, encoder_dim):
        super(ResidualStack, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, encoder_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class vq_conversion(nn.Module):
    def __init__(self, in_feature, out_features):
        super(vq_conversion, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_feature, out_features)
        )

    def forward(self, x):
        return self.layer(x)
    

class vq_pre_residual_stack_decoder(nn.Module):
    def __init__(self, num_embeddings, encoder_dim, dropout):
        super(vq_pre_residual_stack_decoder, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(num_embeddings, encoder_dim),
            nn.GELU(),
            nn.Dropout(dropout)
            )
            
    def forward(self, x):
        return self.layer(x)

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
            
            self.calculated_length = self.find_calculated_length()

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
    



