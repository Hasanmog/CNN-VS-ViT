"""
This code contains some attention mechanism that might be used in architectures
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding2D, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach().requires_grad_(True)
        return x

class Self_Attention(nn.Module):
    def __init__(self, d_model: int):
        super(Self_Attention, self).__init__()
        self.d_model = d_model
        self.transform = nn.Linear(d_model, d_model * 3)  # single transformation
        self.softmax = nn.Softmax(dim=-1)
        self.pos_enc = PositionalEncoding2D(d_model)

    def forward(self, x):
        batch, channels, height, width = x.size()
        x = x.view(batch, channels, height * width).transpose(1, 2)

        # Add positional encoding
        x = self.pos_enc(x)

        # single transformation followed by splitting
        queries, keys, values = self.transform(x).chunk(3, dim=-1)

        scores = torch.einsum('bqd,bkd->bqk', [queries, keys]) / (self.d_model ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.einsum('bqk,bkd->bqd', [attention, values])

        weighted = weighted.transpose(1, 2).view(batch, channels, height, width)
        return weighted
        

