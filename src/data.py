import os 
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Literal
import pickle

class StockDataset(Dataset):
    def __init__(self, seq_len=16, mode: Literal["train", "valid", "test"] = "train"):
        dataset_path = "dataset/NASDAQ"
        test_count = 273
        val_count = 252
        self.seq_len = seq_len
        
        # load data
        with open(os.path.join(dataset_path, "eod_data.pkl"), "rb") as f:
            self.data = pickle.load(f)
        with open(os.path.join(dataset_path, "mask_data.pkl"), "rb") as f:
            self.mask_data = pickle.load(f)
        self.stock_dim = self.data.shape[0]
        self.feature_dim = self.data.shape[-1]
        self._preprocessing()
        
        if mode == "train":
            self.eod_data = self.eod_data[:-test_count-val_count] 
            self.mask_data = self.mask_data[:-test_count-val_count]
            self.target = self.target[:-test_count-val_count]
            self.base_price = self.base_price[:-test_count-val_count]
        elif mode == "valid":
            self.eod_data = self.eod_data[-test_count-val_count:-test_count] 
            self.mask_data = self.mask_data[-test_count-val_count:-test_count]
            self.target = self.target[-test_count-val_count:-test_count]
            self.base_price = self.base_price[-test_count-val_count:-test_count]
        elif mode == "test":
            self.eod_data = self.eod_data[-test_count:]
            self.mask_data = self.mask_data[-test_count:]
            self.target = self.target[-test_count:]
            self.base_price = self.base_price[-test_count:]
            
    def _slice_data(self, data):
        data_counts = data.shape[1] - self.seq_len + 1
        sliced_data = np.array([data[:, i:i+self.seq_len] for i in range(data_counts)])
        return sliced_data
    
    def _preprocessing(self):
        self.eod_data = torch.from_numpy(self._slice_data(self.data)[:-1]).float() # (T, N, L, F)
        self.mask_data = torch.from_numpy(self._slice_data(self.mask_data).min(axis=2)[1:]).float().unsqueeze(-1) # (T, N, 1)
        self.base_price = self.eod_data[:,:, -1, -1].unsqueeze(-1) # (T, N, 1)
        next_price = torch.from_numpy(self._slice_data(self.data)[1:,:,-1,-1]).float().unsqueeze(-1) # (T, N, 1)
        self.target = (next_price - self.base_price) / self.base_price # (T, N, 1)
            
    def __len__(self):
        return self.eod_data.shape[0]

    def __getitem__(self, idx):
        return self.eod_data[idx], self.mask_data[idx], self.base_price[idx], self.target[idx]