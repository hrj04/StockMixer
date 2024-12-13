import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange

class IndicatorMixing(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm((seq_len, feature_dim))
        self.W1 = nn.Parameter(torch.randn(feature_dim, hidden_dim))
        self.act = nn.Hardswish()
        self.W2 = nn.Parameter(torch.randn(hidden_dim, feature_dim))
        
    def forward(self, x):
        h = self.ln(x) # (N, T, F)
        h = h @ self.W1 # (N, T, H)
        h = self.act(h)
        h = h @ self.W2 # (N, T, F)
        out = x + h
        return out # (N, T, F)
        
class TimeMixing(nn.Module):
    def __init__(self, feature_dim, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.ln = nn.LayerNorm((seq_len, feature_dim))
        self.U1 = nn.ParameterList(
            [nn.Parameter(torch.randn(1, i+1)*0.01) for i in range(seq_len)]
        )
        self.act = nn.Hardswish()
        self.U2 = nn.ParameterList(
            [nn.Parameter(torch.randn(1, i+1)*0.01) for i in range(seq_len)]
        )
        
    def forward(self, x):
        h = self.ln(x) # (N, T, F)

        u = self.U1[0] @ h[:, 0:1, :] # (N, 1, F)
        for i in range(1, self.seq_len):
            u = torch.concat([u, self.U1[i] @ h[:, 0:i+1, :]], dim=1) # (N, T, F)
        u = self.act(u)
        
        v = self.U2[0] @ u[:, 0:1, :] # (N, 1, F)
        for i in range(1, self.seq_len):
            v = torch.concat([v, self.U2[i] @ u[:, 0:i+1, :]], dim=1) # (N, T, F)
        
        out = x + v
        return out

class StockMixing(nn.Module):
    def __init__(self, market_dim, stock_num, mix_dim):
        super().__init__()
        self.ln = nn.LayerNorm((stock_num, mix_dim))
        self.M1 = nn.Parameter(torch.randn(market_dim, stock_num))
        self.act = nn.Hardswish()
        self.M2 = nn.Parameter(torch.randn(stock_num, market_dim))
        self.fc = nn.Linear(mix_dim * 2, 1)
        
    def forward(self, x):
        h = self.ln(x) # (N, mix_dim)
        h = self.M1 @ h # (M, mix_dim)
        h = self.act(h)
        h = self.M2 @ h # (N, mix_dim)
        out = x + h
        out = torch.concat([x, out], dim=1)
        out = self.fc(out)
        return out

class IndicatorTimeMixing(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim):
        super().__init__()
        self.indicator_mixing = IndicatorMixing(feature_dim, seq_len, hidden_dim)
        self.time_mixing = TimeMixing(feature_dim, seq_len)
        self.fc = nn.Linear(feature_dim, 1)
        
    def forward(self, x):
        x = self.indicator_mixing(x)
        x = self.time_mixing(x)
        x = self.fc(x)
        x.squeeze_(-1)
        return x
    
class StockMixer(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim, market_dim, stock_num, scale_factors : List):
        super().__init__()
        self.mix_dim = sum([int(seq_len/k) for k in scale_factors])
        self.indicator_time_mixing = nn.ModuleList([
            IndicatorTimeMixing(feature_dim, int(seq_len/k), hidden_dim) for k in scale_factors
        ])
        self.avg1d = nn.ModuleList([
            nn.AvgPool1d(kernel_size=k, stride=k) for k in scale_factors
        ])
        self.stock_mixing = StockMixing(market_dim, stock_num, self.mix_dim)
        
    def forward(self, x):
        mix_tensor = []
        for a, m in zip(self.avg1d, self.indicator_time_mixing):
            x_T = rearrange(x, 'N T F -> N F T')
            x_T = a(x_T)
            x_T = rearrange(x_T, 'N F T -> N T F')
            x_T = m(x_T) # (N, seq_dim/k)
            mix_tensor.append(x_T)
        h = torch.concat(mix_tensor, dim=1) #(N, mix_dim)
        out = self.stock_mixing(h)
        return out

class StockMixerWithConv(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim, market_dim, stock_num, scale_factors : List):
        super().__init__()
        self.mix_dim = sum([int(seq_len/k) for k in scale_factors])
        self.indicator_time_mixing = nn.ModuleList([
            IndicatorTimeMixing(feature_dim, int(seq_len/k), hidden_dim) for k in scale_factors
        ])
        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=k, stride=k) for k in scale_factors
        ])
        self.stock_mixing = StockMixing(market_dim, stock_num, self.mix_dim)
        
    def forward(self, x):
        mix_tensor = []
        for a, m in zip(self.conv1d, self.indicator_time_mixing):
            x_T = rearrange(x, 'N T F -> N F T')
            x_T = a(x_T)
            x_T = rearrange(x_T, 'N F T -> N T F')
            x_T = m(x_T) # (N, seq_dim/k)
            mix_tensor.append(x_T)
        h = torch.concat(mix_tensor, dim=1) #(N, mix_dim)
        out = self.stock_mixing(h)
        return out
