from torch import nn

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
