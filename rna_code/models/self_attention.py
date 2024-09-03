import torch
from torch import nn


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
    
