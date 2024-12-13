import random
import numpy as np
import os
import torch as torch
from load_data import load_EOD_data
from evaluator import evaluate
from model import get_loss, StockMixer
import pickle
from utils import set_seed, get_data, get_batch
from tqdm import tqdm

seed = 123456789
set_seed(seed)
device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
steps = 1

eod_data, mask_data, gt_data, price_data = get_data('NASDAQ', steps)
stock_num = eod_data.shape[0]
trade_dates = mask_data.shape[1]
fea_num = eod_data.shape[2]

lookback_length = 16
valid_index = 756
test_index = 1008

activation = 'GELU'
epochs = 100
scale_factor = 3
alpha = 0.1
market_num = 20


model = StockMixer(
    stocks=stock_num,
    time_steps=lookback_length,
    channels=fea_num,
    market=market_num,
    scale=scale_factor
).to(device)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_valid_loss = np.inf
best_valid_perf = None
best_test_perf = None
batch_offsets = np.arange(start=0, stop=valid_index, dtype=int)

def validate(start_index, end_index):
    with torch.no_grad():
        cur_valid_pred = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_gt = np.zeros([stock_num, end_index - start_index], dtype=float)
        cur_valid_mask = np.zeros([stock_num, end_index - start_index], dtype=float)
        loss = 0.
        reg_loss = 0.
        rank_loss = 0.
        for cur_offset in range(start_index - lookback_length - steps + 1, end_index - lookback_length - steps + 1):
            data_batch, mask_batch, price_batch, gt_batch = map(
                lambda x: torch.Tensor(x).to(device),
                get_batch(eod_data, mask_data, price_data, gt_data, valid_index, lookback_length, steps, cur_offset)
            )
            prediction = model(data_batch)
            cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                                     stock_num, alpha)
            loss += cur_loss.item()
            reg_loss += cur_reg_loss.item()
            rank_loss += cur_rank_loss.item()
            cur_valid_pred[:, cur_offset - (start_index - lookback_length - steps + 1)] = cur_rr[:, 0].cpu()
            cur_valid_gt[:, cur_offset - (start_index - lookback_length - steps + 1)] = gt_batch[:, 0].cpu()
            cur_valid_mask[:, cur_offset - (start_index - lookback_length - steps + 1)] = mask_batch[:, 0].cpu()
        loss = loss / (end_index - start_index)
        reg_loss = reg_loss / (end_index - start_index)
        rank_loss = rank_loss / (end_index - start_index)
        cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt, cur_valid_mask)
    return loss, reg_loss, rank_loss, cur_valid_perf

for epoch in tqdm(range(epochs)):
    np.random.shuffle(batch_offsets)
    tra_loss = 0.0
    tra_reg_loss = 0.0
    tra_rank_loss = 0.0
    data_counts = valid_index - lookback_length - steps + 1
    for j in range(data_counts):
        data_batch, mask_batch, price_batch, gt_batch = map(
            lambda x: torch.Tensor(x).to(device),
            get_batch(eod_data, mask_data, price_data, gt_data, valid_index, lookback_length, steps, batch_offsets[j])
        )
        optimizer.zero_grad()
        prediction = model(data_batch)
        cur_loss, cur_reg_loss, cur_rank_loss, _ = get_loss(prediction, gt_batch, price_batch, mask_batch,
                                                            stock_num, alpha)
        cur_loss = cur_loss
        cur_loss.backward()
        optimizer.step()

        tra_loss += cur_loss.item()
        tra_reg_loss += cur_reg_loss.item()
        tra_rank_loss += cur_rank_loss.item()
    tra_loss = tra_loss / data_counts
    tra_reg_loss = tra_reg_loss / data_counts
    tra_rank_loss = tra_rank_loss / data_counts
    print(f'Train : loss:{tra_loss:.4f}  =  {tra_reg_loss:.4f} + alpha*{tra_rank_loss:.4f}')

    val_loss, val_reg_loss, val_rank_loss, val_perf = validate(valid_index, test_index)
    print(f'Valid : loss:{val_loss:.4f}  =  {val_reg_loss:.4f} + alpha*{val_rank_loss:.4f}')
    
    test_loss, test_reg_loss, test_rank_loss, test_perf = validate(test_index, trade_dates)
    print(f'Test: loss:{test_loss:.4f}  =  {test_reg_loss:.4f} + alpha*{test_rank_loss:.4f}')

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        best_valid_perf = val_perf
        best_test_perf = test_perf
    
    print(f'Best Valid performance: mse:{best_valid_perf["mse"]:.4f} IC:{best_valid_perf["IC"]:.4f} RIC:{best_valid_perf["RIC"]:.4f} prec@10:{best_valid_perf["prec_10"]:.4f}, SR:{best_valid_perf["sharpe5"]:.4f}')
    print(f'Best Test performance: mse:{best_test_perf["mse"]:.4f} IC:{best_test_perf["IC"]:.4f} RIC:{best_test_perf["RIC"]:.4f} prec@10:{best_test_perf["prec_10"]:.4f}, SR:{best_test_perf["sharpe5"]:.4f}\n\n')
