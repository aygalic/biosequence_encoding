from torch import nn
    
class vq_conversion(nn.Module):
    def __init__(self, in_feature, out_features):
        super(vq_conversion, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(in_feature, out_features)
        )

    def forward(self, x):
        return self.layer(x)
    