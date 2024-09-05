from torch import nn


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
