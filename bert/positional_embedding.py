import torch
import math

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        # Create matrix of (seq_length, hidden_size)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
        
    def to(self, device):
        super().to(device)
        self.pe = self.pe.to(device)
        return self 