import torch
import numpy as np

from tqdm import tqdm
from einops import rearrange
from src.models import get_loss, StockMixer, LSTM, Linear, \
    StockMixerWithOutIndicatorMixing, StockMixerWithOutStockMixing, StockMixerWithOutTimeMixing
from src.utils import evaluate
from src.data import StockDataset
from torch.utils.data import DataLoader
from typing import List, Literal


def StockMixer_Table2(seq_len, 
                      hidden_dim, 
                      market_dim, 
                      scale_factors: List, 
                      alpha, epochs, 
                      seed, 
                      load_weights=False
                      ):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    weights_path = f"weights/stockmixer_L{seq_len}_H{hidden_dim}_M{market_dim}_K{len(scale_factors)}_E{epochs}_s{seed}.pt"

    # load data
    train_dataset = StockDataset(seq_len=seq_len, mode="train")
    test_dataset = StockDataset(seq_len=seq_len, mode="test")
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # model parameters
    stock_num = train_dataset.stock_dim
    feature_dim = train_dataset.feature_dim
    learning_rate = 0.001
    
    model = StockMixer(feature_dim, seq_len, hidden_dim, market_dim, stock_num, scale_factors).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if load_weights:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
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
        return test_performance
    else:
        best_test_RIC = -np.inf
        best_test_perf = None
        pbar = tqdm(range(epochs), desc="Epoch")
        for epoch in pbar:
            # train
            model.train()
            for data_batch, mask_batch, base_batch, target_batch in train_dl:
                data_batch, mask_batch, base_batch, target_batch = data_batch.squeeze(0).to(device), mask_batch.squeeze(0).to(device), base_batch.squeeze(0).to(device), target_batch.squeeze(0).to(device)
                optimizer.zero_grad()
                prediction = model(data_batch)
                loss, *_ = get_loss(prediction, target_batch, base_batch, mask_batch, alpha)
                loss.backward()
                optimizer.step()
                
            # realtime test
            model.eval()
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
                torch.save(model.state_dict(), weights_path)
            pbar.set_postfix(test_loss=test_performance['mse'])
        return best_test_perf


def LSTM_Table2(seq_len, hidden_dim, alpha, epochs, seed, load_weights=False):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    weights_path = f"weights/LSTM_L{seq_len}_H{hidden_dim}_E{epochs}_s{seed}.pt"

    # load data
    train_dataset = StockDataset(seq_len=seq_len, mode="train")
    test_dataset = StockDataset(seq_len=seq_len, mode="test")
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # model parameters
    feature_dim = train_dataset.feature_dim
    learning_rate = 0.001
    
    model = LSTM(feature_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if load_weights:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
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
        return test_performance
    else:
        best_test_RIC = -np.inf
        best_test_perf = None
        pbar = tqdm(range(epochs), desc="Epoch")
        for epoch in pbar:
            # train
            model.train()
            for data_batch, mask_batch, base_batch, target_batch in train_dl:
                data_batch, mask_batch, base_batch, target_batch = data_batch.squeeze(0).to(device), mask_batch.squeeze(0).to(device), base_batch.squeeze(0).to(device), target_batch.squeeze(0).to(device)
                optimizer.zero_grad()
                prediction = model(data_batch)
                loss, *_ = get_loss(prediction, target_batch, base_batch, mask_batch, alpha)
                loss.backward()
                optimizer.step()
                
            # realtime test
            model.eval()
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
                torch.save(model.state_dict(), weights_path)
            pbar.set_postfix(test_loss=test_performance['mse'])
        return best_test_perf


def Linear_Table2(seq_len, hidden_dim, alpha, epochs, seed, load_weights=False):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    weights_path = f"weights/Linear_L{seq_len}_H{hidden_dim}_E{epochs}_s{seed}.pt"

    # load data
    train_dataset = StockDataset(seq_len=seq_len, mode="train")
    test_dataset = StockDataset(seq_len=seq_len, mode="test")
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # model parameters
    feature_dim = train_dataset.feature_dim
    learning_rate = 0.001
    input_dim = feature_dim * seq_len
    model = Linear(input_dim, hidden_dim, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if load_weights:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
        test_pred = []
        for i in range(len(test_dataset)):
            data_batch = test_dataset.eod_data[i].view(-1,input_dim).to(device)
            base_batch = test_dataset.base_price[i].to(device)
            prediction = model(data_batch)
            target = (prediction - base_batch) / base_batch
            test_pred.append(target.cpu().detach().numpy())
        test_pred = rearrange(test_pred, "T N 1 -> N T")
        target = rearrange(test_dataset.target, "T N 1 -> N T").numpy()
        mask = rearrange(test_dataset.mask_data, "T N 1 -> N T").numpy()
        test_performance = evaluate(test_pred, target, mask)
        return test_performance
    else:
        best_test_RIC = -np.inf
        best_test_perf = None
        pbar = tqdm(range(epochs), desc="Epoch")
        for epoch in pbar:
            # train
            model.train()
            for data_batch, mask_batch, base_batch, target_batch in train_dl:
                data_batch, mask_batch, base_batch, target_batch = data_batch.squeeze(0).view(-1,input_dim).to(device), mask_batch.squeeze(0).to(device), base_batch.squeeze(0).to(device), target_batch.squeeze(0).to(device)
                optimizer.zero_grad()
                prediction = model(data_batch)
                loss, *_ = get_loss(prediction, target_batch, base_batch, mask_batch, alpha)
                loss.backward()
                optimizer.step()
                
            # realtime test
            model.eval()
            test_pred = []
            for i in range(len(test_dataset)):
                data_batch = test_dataset.eod_data[i].view(-1,input_dim).to(device)
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
                torch.save(model.state_dict(), weights_path)
            pbar.set_postfix(test_loss=test_performance['mse'])
        return best_test_perf


def StockMixer_Table3(seq_len, 
                      hidden_dim, 
                      market_dim,
                      scale_factors: List, 
                      alpha, 
                      epochs, 
                      seed, 
                      load_weights=False
                      ):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    weights_wo_indicator_path = f"weights/stockmixer_wo_indicator_L{seq_len}_H{hidden_dim}_M{market_dim}_K{len(scale_factors)}_E{epochs}_s{seed}.pt"
    weights_wo_time_path = f"weights/stockmixer_wo_time_L{seq_len}_H{hidden_dim}_M{market_dim}_K{len(scale_factors)}_E{epochs}_s{seed}.pt"
    weights_wo_stock_path = f"weights/stockmixer_wo_stock_L{seq_len}_H{hidden_dim}_M{market_dim}_K{len(scale_factors)}_E{epochs}_s{seed}.pt"

    # load data
    train_dataset = StockDataset(seq_len=seq_len, mode="train")
    test_dataset = StockDataset(seq_len=seq_len, mode="test")
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # model parameters
    stock_num = train_dataset.stock_dim
    feature_dim = train_dataset.feature_dim
    learning_rate = 0.001
    model_wo_indicator = StockMixerWithOutIndicatorMixing(feature_dim,
                                                          seq_len, 
                                                          market_dim, 
                                                          stock_num,
                                                          scale_factors).to(device)
    model_wo_time = StockMixerWithOutTimeMixing(feature_dim,
                                                seq_len,
                                                hidden_dim,
                                                market_dim,
                                                stock_num,
                                                scale_factors).to(device)
    model_wo_stock = StockMixerWithOutStockMixing(feature_dim, 
                                                  seq_len, 
                                                  hidden_dim, 
                                                  scale_factors).to(device)
    optimizer_wo_indicator = torch.optim.Adam(model_wo_indicator.parameters(), lr=learning_rate)
    optimizer_wo_time = torch.optim.Adam(model_wo_time.parameters(), lr=learning_rate)
    optimizer_wo_stock = torch.optim.Adam(model_wo_stock.parameters(), lr=learning_rate)
    
    if load_weights:
        model_wo_indicator.load_state_dict(torch.load(weights_wo_indicator_path, weights_only=True))
        model_wo_time.load_state_dict(torch.load(weights_wo_time_path, weights_only=True))
        model_wo_stock.load_state_dict(torch.load(weights_wo_stock_path, weights_only=True))
        
        model_wo_indicator.eval()
        model_wo_time.eval()
        model_wo_stock.eval()
        
        test_pred_wo_indicator = []
        test_pred_wo_time = []
        test_pred_wo_stock = []
        for i in range(len(test_dataset)):
            data_batch = test_dataset.eod_data[i].to(device)
            base_batch = test_dataset.base_price[i].to(device)
            prediction_wo_indicator = model_wo_indicator(data_batch)
            prediction_wo_time = model_wo_time(data_batch)
            prediction_wo_stock = model_wo_stock(data_batch)
            
            target_wo_indicator = (prediction_wo_indicator - base_batch) / base_batch
            target_wo_time = (prediction_wo_time - base_batch) / base_batch
            target_wo_stock = (prediction_wo_stock - base_batch) / base_batch
            
            test_pred_wo_indicator.append(target_wo_indicator.cpu().detach().numpy())
            test_pred_wo_time.append(target_wo_time.cpu().detach().numpy())
            test_pred_wo_stock.append(target_wo_stock.cpu().detach().numpy())
        
        test_pred_wo_indicator = rearrange(test_pred_wo_indicator, "T N 1 -> N T")
        test_pred_wo_time = rearrange(test_pred_wo_time, "T N 1 -> N T")
        test_pred_wo_stock = rearrange(test_pred_wo_stock, "T N 1 -> N T")
        target = rearrange(test_dataset.target, "T N 1 -> N T").numpy()
        mask = rearrange(test_dataset.mask_data, "T N 1 -> N T").numpy()
        
        test_performance_wo_indicator = evaluate(test_pred_wo_indicator, target, mask)
        test_performance_wo_time = evaluate(test_pred_wo_time, target, mask)
        test_performance_wo_stock = evaluate(test_pred_wo_stock, target, mask)
        
        return test_performance_wo_indicator, test_performance_wo_time, test_performance_wo_stock
    else:
        best_test_wo_indicator_RIC = -np.inf
        best_test_wo_time_RIC = -np.inf
        best_test_wo_stock_RIC = -np.inf
        
        best_test_wo_indicator_performance = None
        best_test_wo_time_performance = None
        best_test_wo_stock_performance = None
        
        pbar = tqdm(range(epochs), desc="Epoch")
        for epoch in pbar:
            # train
            model_wo_indicator.train()
            model_wo_time.train()
            model_wo_stock.train()
            for data_batch, mask_batch, base_batch, target_batch in train_dl:
                data_batch, mask_batch, base_batch, target_batch = data_batch.squeeze(0).to(device), mask_batch.squeeze(0).to(device), base_batch.squeeze(0).to(device), target_batch.squeeze(0).to(device)
                optimizer_wo_indicator.zero_grad()
                optimizer_wo_time.zero_grad()
                optimizer_wo_stock.zero_grad()
                
                prediction_wo_indicator = model_wo_indicator(data_batch)
                prediction_wo_time = model_wo_time(data_batch)
                prediction_wo_stock = model_wo_stock(data_batch)
                
                loss_wo_indicator, *_ = get_loss(prediction_wo_indicator, target_batch, base_batch, mask_batch, alpha)
                loss_wo_time, *_ = get_loss(prediction_wo_time, target_batch, base_batch, mask_batch, alpha)
                loss_wo_stock, *_ = get_loss(prediction_wo_stock, target_batch, base_batch, mask_batch, alpha)
                
                loss_wo_indicator.backward()
                loss_wo_time.backward()
                loss_wo_stock.backward()
                
                optimizer_wo_indicator.step()
                optimizer_wo_time.step()
                optimizer_wo_stock.step()
                                
            # realtime test
            model_wo_indicator.eval()
            model_wo_time.eval()
            model_wo_stock.eval()
            
            test_pred_wo_indicator = []
            test_pred_wo_time = []
            test_pred_wo_stock = []
            for i in range(len(test_dataset)):
                data_batch = test_dataset.eod_data[i].to(device)
                base_batch = test_dataset.base_price[i].to(device)
                prediction_wo_indicator = model_wo_indicator(data_batch)
                prediction_wo_time = model_wo_time(data_batch)
                prediction_wo_stock = model_wo_stock(data_batch)
                
                target_wo_indicator = (prediction_wo_indicator - base_batch) / base_batch
                target_wo_time = (prediction_wo_time - base_batch) / base_batch
                target_wo_stock = (prediction_wo_stock - base_batch) / base_batch
                
                test_pred_wo_indicator.append(target_wo_indicator.cpu().detach().numpy())
                test_pred_wo_time.append(target_wo_time.cpu().detach().numpy())
                test_pred_wo_stock.append(target_wo_stock.cpu().detach().numpy())
            
            test_pred_wo_indicator = rearrange(test_pred_wo_indicator, "T N 1 -> N T")
            test_pred_wo_time = rearrange(test_pred_wo_time, "T N 1 -> N T")
            test_pred_wo_stock = rearrange(test_pred_wo_stock, "T N 1 -> N T")
            target = rearrange(test_dataset.target, "T N 1 -> N T").numpy()
            mask = rearrange(test_dataset.mask_data, "T N 1 -> N T").numpy()
            
            test_performance_wo_indicator = evaluate(test_pred_wo_indicator, target, mask)
            test_performance_wo_time = evaluate(test_pred_wo_time, target, mask)
            test_performance_wo_stock = evaluate(test_pred_wo_stock, target, mask)
            
            if best_test_wo_indicator_RIC < test_performance_wo_indicator["RIC"]:
                best_test_wo_indicator_RIC = test_performance_wo_indicator["RIC"]
                best_test_wo_indicator_performance = test_performance_wo_indicator
                torch.save(model_wo_indicator.state_dict(), weights_wo_indicator_path)
            if best_test_wo_time_RIC < test_performance_wo_time["RIC"]:
                best_test_wo_time_RIC = test_performance_wo_time["RIC"]
                best_test_wo_time_performance = test_performance_wo_time
                torch.save(model_wo_time.state_dict(), weights_wo_time_path)
            if best_test_wo_stock_RIC < test_performance_wo_stock["RIC"]:
                best_test_wo_stock_RIC = test_performance_wo_stock["RIC"]
                best_test_wo_stock_performance = test_performance_wo_stock
                torch.save(model_wo_stock.state_dict(), weights_wo_stock_path)
                
            pbar.set_postfix(test_loss_wo_indicator=test_performance_wo_indicator['mse'], 
                             test_loss_wo_time=test_performance_wo_time['mse'],
                             test_loss_wo_stock=test_performance_wo_stock['mse'])
        return best_test_wo_indicator_performance, best_test_wo_time_performance, best_test_wo_stock_performance


def StockMixer_Figure3(seq_len, 
                       hidden_dim, 
                       market_dim, 
                       scale_factors: List, 
                       alpha, 
                       epochs, 
                       seed, 
                       act: Literal["Hardswish", "ReLU", "GELU", "Sigmoid", "tanh"] = "Hardswish", 
                       load_weights=False):
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    weights_path = f"weights/stockmixer_L{seq_len}_H{hidden_dim}_M{market_dim}_K{len(scale_factors)}_E{epochs}_s{seed}_{act}.pt"

    # load data
    train_dataset = StockDataset(seq_len=seq_len, mode="train")
    test_dataset = StockDataset(seq_len=seq_len, mode="test")
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # model parameters
    stock_num = train_dataset.stock_dim
    feature_dim = train_dataset.feature_dim
    learning_rate = 0.001
    
    model = StockMixer(feature_dim, 
                       seq_len, 
                       hidden_dim, 
                       market_dim, 
                       stock_num, 
                       scale_factors,
                       act
                       ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if load_weights:
        model.load_state_dict(torch.load(weights_path, weights_only=True))
        model.eval()
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
        return test_performance
    else:
        best_test_RIC = -np.inf
        best_test_perf = None
        pbar = tqdm(range(epochs), desc="Epoch")
        for epoch in pbar:
            # train
            model.train()
            for data_batch, mask_batch, base_batch, target_batch in train_dl:
                data_batch, mask_batch, base_batch, target_batch = data_batch.squeeze(0).to(device), mask_batch.squeeze(0).to(device), base_batch.squeeze(0).to(device), target_batch.squeeze(0).to(device)
                optimizer.zero_grad()
                prediction = model(data_batch)
                loss, *_ = get_loss(prediction, target_batch, base_batch, mask_batch, alpha)
                loss.backward()
                optimizer.step()
                
            # realtime test
            model.eval()
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
                torch.save(model.state_dict(), weights_path)
            pbar.set_postfix(test_loss=test_performance['mse'])
        return best_test_perf
