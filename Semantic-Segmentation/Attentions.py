"""
This code contains some attention mechanism that might be used in architectures
"""
import torch
import torch.nn as nn


class Self_Attention(nn.Module):
    def __init__(self , d_model : int ):
        super(Self_Attention, self).__init__()
        self.d_model = d_model
        self.query = nn.Linear(d_model , d_model )
        self.key = nn.Linear(d_model , d_model )
        self.value = nn.Linear(d_model , d_model )
        self.softmax = nn.Softmax(dim = 2)
        
    def forward(self , x):
        # print("size" , x.shape)
        batch , channels , height , width = x.size()
        
        x = x.view(batch, channels, height * width).transpose(1,2)
        
        queries  = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # print("Queries shape:", queries.shape)
        # print("Keys shape:", keys.shape)
        # print("Values shape:", values.shape)

        
        scores = torch.bmm(queries, keys.transpose(1,2)) / (self.d_model ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention , values)
        
        weighted = weighted.transpose(1,2).view(batch , channels , height , width)
        
        return weighted
        

