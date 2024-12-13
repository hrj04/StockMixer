import os
import sys
from models.StockMixer import get_loss
from models.StockMixer2 import StockMixerWithConv
from einops import rearrange
from utils import set_seed, evaluate
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import StockDataset

seeds = [1, 2, 3, 4, 5]
mse_lst = []
IC_lst = []
RIC_lst = []
sharpe5_lst = []
prec_10_lst = []

for seed in seeds:
    set_seed(seed)

    # load data
    train_dataset = StockDataset(seq_len=16, mode="train")
    val_dataset = StockDataset(seq_len=16, mode="valid")
    test_dataset = StockDataset(seq_len=16, mode="test")

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    stock_num = train_dataset.stock_dim
    feature_dim = train_dataset.feature_dim
    seq_len = train_dataset.seq_len
    hidden_dim = 64
    market_dim = 20
    scale_factors = [1, 2, 4, 8]

    model = StockMixerWithConv(feature_dim, seq_len, hidden_dim, market_dim, stock_num, scale_factors).to(device)
    alpha = 0.1
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = 50
    best_test_RIC = -np.inf
    best_valid_perf = None
    best_test_perf = None
    
    pbar = tqdm(range(epochs), desc="Epoch")
    for epoch in pbar:
        model.train()
        for data_batch, mask_batch, base_batch, target_batch in train_dl:
            data_batch, mask_batch, base_batch, target_batch = data_batch.squeeze(0).to(device), mask_batch.squeeze(0).to(device), base_batch.squeeze(0).to(device), target_batch.squeeze(0).to(device)
            optimizer.zero_grad()
            prediction = model(data_batch)
            loss, reg_loss, rank_loss, _ = get_loss(prediction, target_batch, base_batch, mask_batch, alpha)
            loss.backward()
            optimizer.step()
            
        model.eval()
        # validate
        val_pred = []
        for i in range(len(val_dataset)):
            data_batch = val_dataset.eod_data[i].to(device)
            base_batch = val_dataset.base_price[i].to(device)
            prediction = model(data_batch)
            target = (prediction - base_batch) / base_batch
            val_pred.append(target.cpu().detach().numpy())
            
        val_pred = rearrange(val_pred, "T N 1 -> N T")
        target = rearrange(val_dataset.target, "T N 1 -> N T").numpy()
        mask = rearrange(val_dataset.mask_data, "T N 1 -> N T").numpy()
        val_performance = evaluate(val_pred, target, mask)

        # test
        test_pred = []
        for i in range(len(test_dataset)):
            data_batch = test_dataset.eod_data[i].to(device)
            base_batch = test_dataset.base_price[i].to(device)
            prediction = model(data_batch)
            target = (prediction - base_batch) / base_batch
            test_pred.append(target.cpu().detach().numpy())
        test_pred = rearrange(test_pred, "T N 1 -> N T")
        target = rearrange(test_dataset.target, "T N 1 -> N T").numpy()
        mask = rearrange(test_dataset.mask_data, "T N 1 -> N T").numpy()
        test_performance = evaluate(test_pred, target, mask)
        
        if best_test_RIC < test_performance["RIC"]:
            best_test_RIC = test_performance["RIC"]
            best_test_perf = test_performance
            torch.save(model.state_dict(), "lstm.pt")
            
        pbar.set_postfix(val_loss=val_performance['mse'], test_loss=test_performance['mse'])
        print("\n")
        print(f"Epoch : {epoch+1}, val_mse : {val_performance['mse']:0.4f}, val_IC : {val_performance['IC']:0.4f}, val_RIC : {val_performance['RIC']:0.4f}, val_sharpe5 : {val_performance['sharpe5']:0.4f}, val_prec_10 : {val_performance['prec_10']:0.4f}")
        print(f"Epoch : {epoch+1}, test_mse : {best_test_perf['mse']:0.4f}, test_IC : {best_test_perf['IC']:0.4f}, test_RIC : {best_test_perf['RIC']:0.4f}, test_sharpe5 : {best_test_perf['sharpe5']:0.4f}, test_prec_10 : {best_test_perf['prec_10']:0.4f}")
        sys.stdout.flush()
    mse_lst.append(best_test_perf['mse'])
    IC_lst.append(best_test_perf['IC'])
    RIC_lst.append(best_test_perf['RIC'])
    sharpe5_lst.append(best_test_perf['sharpe5'])
    prec_10_lst.append(best_test_perf['prec_10'])

print("\n")
print(f"mse : {np.mean(mse_lst):0.4f} +- {np.std(mse_lst):0.4f}")
print(f"IC : {np.mean(IC_lst):0.4f} +- {np.std(IC_lst):0.4f}")
print(f"RIC : {np.mean(RIC_lst):0.4f} +- {np.std(RIC_lst):0.4f}")
print(f"sharpe5 : {np.mean(sharpe5_lst):0.4f} +- {np.std(sharpe5_lst):0.4f}")
print(f"prec_10 : {np.mean(prec_10_lst):0.4f} +- {np.std(prec_10_lst):0.4f}")
sys.stdout.flush()