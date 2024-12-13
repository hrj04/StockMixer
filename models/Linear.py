import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, hidden_dim)
        self.activation = nn.Hardswish()
        self.ln = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, out_features)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.ln(x)
        x = self.layer2(x)
        return x
    
    