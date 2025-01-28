import torch
import torch.nn.functional as F
import math

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask):
        query = self.query(query)
        key = self.key(key)        
        value = self.value(value)   
        
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)   
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)  
        
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(mask == 0, -1e9)    
        weights = F.softmax(scores, dim=-1)           
        weights = self.dropout(weights)
        context = torch.matmul(weights, value)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        return self.output_linear(context)

    def to(self, device):
        super().to(device)
        self.query = self.query.to(device)
        self.key = self.key.to(device)
        self.value = self.value.to(device)
        self.output_linear = self.output_linear.to(device)
        return self 