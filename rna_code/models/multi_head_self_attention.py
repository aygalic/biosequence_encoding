import torch.nn as nn
import torch.nn.functional as F
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