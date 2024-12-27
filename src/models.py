import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Literal
from einops import rearrange


def get_loss(prediction, ground_truth, base_price, mask, alpha):
    device = prediction.device
    all_one = torch.ones(prediction.shape[0], 1, dtype=torch.float32).to(device)
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )
    loss = reg_loss + alpha * rank_loss
    return loss, reg_loss, rank_loss, return_ratio


class IndicatorMixing(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 seq_len, 
                 hidden_dim, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"
                 ):
        super().__init__()
        self.ln = nn.LayerNorm((seq_len, feature_dim))
        self.W1 = nn.Parameter(torch.randn(feature_dim, hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, feature_dim))
        
        if act == "Hardswish":
            self.act = nn.Hardswish()
        elif act == "ReLU":
            self.act = nn.ReLU()
        elif act == "GELU":
            self.act = nn.GELU()
        elif act == "Sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        
    def forward(self, x):
        h = self.ln(x) # (N, T, F)
        h = h @ self.W1 # (N, T, H)
        h = self.act(h)
        h = h @ self.W2 # (N, T, F)
        out = x + h
        return out # (N, T, F)
        
class TimeMixing(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 seq_len, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"):
        super().__init__()
        self.seq_len = seq_len
        self.ln = nn.LayerNorm((seq_len, feature_dim))
        self.U1 = nn.ParameterList(
            [nn.Parameter(torch.randn(1, i+1)*0.01) for i in range(seq_len)]
        )
        self.U2 = nn.ParameterList(
            [nn.Parameter(torch.randn(1, i+1)*0.01) for i in range(seq_len)]
        )
        
        if act == "Hardswish":
            self.act = nn.Hardswish()
        elif act == "ReLU":
            self.act = nn.ReLU()
        elif act == "GELU":
            self.act = nn.GELU()
        elif act == "Sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
            
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
    def __init__(self, 
                 market_dim, 
                 stock_num, 
                 mix_dim, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"):
        super().__init__()
        self.ln = nn.LayerNorm((stock_num, mix_dim))
        self.M1 = nn.Parameter(torch.randn(market_dim, stock_num))
        self.M2 = nn.Parameter(torch.randn(stock_num, market_dim))
        self.fc = nn.Linear(mix_dim * 2, 1)
        
        if act == "Hardswish":
            self.act = nn.Hardswish()
        elif act == "ReLU":
            self.act = nn.ReLU()
        elif act == "GELU":
            self.act = nn.GELU()
        elif act == "Sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        
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
    def __init__(self, 
                 feature_dim, 
                 seq_len, 
                 hidden_dim, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"):
        super().__init__()
        self.indicator_mixing = IndicatorMixing(feature_dim, seq_len, hidden_dim, act)
        self.time_mixing = TimeMixing(feature_dim, seq_len, act)
        self.fc = nn.Linear(feature_dim, 1)
        
    def forward(self, x):
        x = self.indicator_mixing(x)
        x = self.time_mixing(x)
        x = self.fc(x)
        x.squeeze_(-1)
        return x

class StockMixer(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 seq_len, 
                 hidden_dim, 
                 market_dim, 
                 stock_num, 
                 scale_factors : List, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"):
        super().__init__()
        self.mix_dim = sum([int(seq_len/k) for k in scale_factors])
        self.indicator_time_mixing = nn.ModuleList([
            IndicatorTimeMixing(feature_dim, int(seq_len/k), hidden_dim, act) for k in scale_factors
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

class StockMixerWithOutIndicatorMixing(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 seq_len, 
                 market_dim, 
                 stock_num, 
                 scale_factors : List, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"):
        super().__init__()
        self.mix_dim = sum([int(seq_len/k) for k in scale_factors])
        self.time_mixing = nn.ModuleList([
            TimeMixing(feature_dim, int(seq_len/k), act) for k in scale_factors
        ])
        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=k, stride=k) for k in scale_factors
        ])
        self.fc = nn.Linear(feature_dim, 1)
        self.stock_mixing = StockMixing(market_dim, stock_num, self.mix_dim)
    
    def forward(self, x):
        mix_tensor = []
        for a, m in zip(self.conv1d, self.time_mixing):
            x_T = rearrange(x, 'N T F -> N F T')
            x_T = a(x_T)
            x_T = rearrange(x_T, 'N F T -> N T F')
            x_T = m(x_T) # (N, seq_dim/k, F)
            x_T = self.fc(x_T).squeeze(-1) # (N, seq_dim/k)
            mix_tensor.append(x_T)
        h = torch.concat(mix_tensor, dim=1) #(N, mix_dim)
        out = self.stock_mixing(h)
        return out
    
class StockMixerWithOutTimeMixing(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 seq_len, 
                 hidden_dim, 
                 market_dim, 
                 stock_num, 
                 scale_factors : List, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"):
        super().__init__()
        self.mix_dim = sum([int(seq_len/k) for k in scale_factors])
        self.indicator_mixing = nn.ModuleList([
            IndicatorMixing(feature_dim, int(seq_len/k), hidden_dim, act) for k in scale_factors
        ])
        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=k, stride=k) for k in scale_factors
        ])
        self.fc = nn.Linear(feature_dim, 1)
        self.stock_mixing = StockMixing(market_dim, stock_num, self.mix_dim)
    def forward(self, x):
        mix_tensor = []
        for a, m in zip(self.conv1d, self.indicator_mixing):
            x_T = rearrange(x, 'N T F -> N F T')
            x_T = a(x_T)
            x_T = rearrange(x_T, 'N F T -> N T F')
            x_T = m(x_T) # (N, seq_dim/k, F)
            x_T = self.fc(x_T).squeeze(-1) # (N, seq_dim/k)
            mix_tensor.append(x_T)
        h = torch.concat(mix_tensor, dim=1) #(N, mix_dim)
        out = self.stock_mixing(h)
        return out

class StockMixerWithOutStockMixing(nn.Module):
    def __init__(self, 
                 feature_dim, 
                 seq_len, 
                 hidden_dim, 
                 scale_factors : List, 
                 act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish"):
        super().__init__()
        self.mix_dim = sum([int(seq_len/k) for k in scale_factors])
        self.indicator_time_mixing = nn.ModuleList([
            IndicatorTimeMixing(feature_dim, int(seq_len/k), hidden_dim, act) for k in scale_factors
        ])
        self.conv1d = nn.ModuleList([
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=k, stride=k) for k in scale_factors
        ])
        self.fc = nn.Linear(self.mix_dim, 1)
    
    def forward(self, x):
        mix_tensor = []
        for a, m in zip(self.conv1d, self.indicator_time_mixing):
            x_T = rearrange(x, 'N T F -> N F T')
            x_T = a(x_T)
            x_T = rearrange(x_T, 'N F T -> N T F')
            x_T = m(x_T) # (N, seq_dim/k)
            mix_tensor.append(x_T)
        h = torch.concat(mix_tensor, dim=1) #(N, mix_dim)
        out = self.fc(h)
        return out 


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x


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
    
