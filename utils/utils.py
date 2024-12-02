import os
import logging
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from .model import MLPClassifier
from .metric import evaluate

def setup_logger(args, label_suffix, idx):
    os.makedirs('results/records', exist_ok=True)
    log_filename = f'results/records/{args.data}_{args.label}_{args.resample}_{args.dim_red}_{args.dim_num}_{args.loss}_{args.model}_{label_suffix}.log'
    logger = logging.getLogger(f'{label_suffix}_{args.model}_{idx}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger

def setup_logger_benchmark(label_suffix, idx):
    os.makedirs('results/records', exist_ok=True)
    log_filename = f'results/records/benchmark_{label_suffix}.log'
    logger = logging.getLogger(f'{label_suffix}_benchmark_{idx}')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger


import joblib
class BPModel:
    def __init__(self, model, dim_red, resample):
        self.model = model
        self.dim_red = dim_red
        self.resample = resample

    def fit_preprocess(self, X, y):
        if self.dim_red is not None:
            X_red = self.dim_red.fit_transform(X)
        else:
            X_red = X

        if self.resample is not None:
            X_res, y_res = self.resample.fit_resample(X_red, y.to_numpy()[:, 0])
        else:
            X_res = X_red

        return X_res, y_res

    def fit(self, X, y): 
        X_preprocessed, y_preprocessed = self.fit_preprocess(X, y)
        return self.model.fit(X_preprocessed, y_preprocessed)
    
    def preprocess(self, X):
        if self.dim_red is not None:
            X_red = self.dim_red.transform(X)
        else:
            X_red = X

        return X_red
    
    def predict(self, X):
        X_preprocessed = self.preprocess(X)
        return self.model.predict(X_preprocessed, return_winning_probability=True)

    def save(self, filename):
        joblib.dump(self, filename)

    @staticmethod
    def load(filename):
        return joblib.load(filename)

def repeat_once(args, seed, data, label, label_name, dim_red, resample, loss):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.4, random_state=seed)

    # Filter
    # Xy_test = pd.concat([X_test, y_test], axis=1)
    # condition1 = (Xy_test.iloc[:, -8] + Xy_test.iloc[:, -5]) / 2 < 130
    # condition2 = (Xy_test.iloc[:, -7] + Xy_test.iloc[:, -4]) / 2 < 80
    # condition = condition1 & condition2
    # X_test = Xy_test[condition].iloc[:, :-2]
    # y_test = Xy_test[condition].iloc[:, -2:]
    # y_rest = Xy_test[~condition].iloc[:, -2:]
    
    if args.dim_red == "t-SNE":
        X_train_red = dim_red[args.dim_red].fit_transform(X_train)
        X_test_red = dim_red[args.dim_red].fit_transform(X_test)
        X_all_red = dim_red[args.dim_red].fit_transform(data)
    elif dim_red[args.dim_red] is not None:
        X_train_red = dim_red[args.dim_red].fit_transform(X_train)
        X_test_red = dim_red[args.dim_red].transform(X_test)
        X_all_red = dim_red[args.dim_red].transform(data)
    else:
        X_train_red = X_train.to_numpy()
        X_test_red = X_test.to_numpy()
        X_all_red = data.to_numpy()

    if resample[args.resample] is not None:
        X_train_res_MH, y_train_res_MH = resample[args.resample].fit_resample(X_train_red, y_train.to_numpy()[:, 0])
        X_train_res_WCH, y_train_res_WCH = resample[args.resample].fit_resample(X_train_red, y_train.to_numpy()[:, 1])
    else:
        X_train_res_MH, y_train_res_MH = X_train_red, y_train.to_numpy()[:, 0]
        X_train_res_WCH, y_train_res_WCH = X_train_red, y_train.to_numpy()[:, 1]
        
    if args.model=="SVM":
        model_MH = SVC(gamma="auto")
        model_WCH = SVC(gamma="auto")
    elif args.model=="MLP":
        model_MH = MLPClassifier(X_train_res_MH.shape[1]).to(device)
        model_WCH = MLPClassifier(X_train_res_WCH.shape[1]).to(device)
    elif args.model=="TabPFN":
        model_MH = TabPFNClassifier(device=device, N_ensemble_configurations=32)
        model_WCH = TabPFNClassifier(device=device, N_ensemble_configurations=32)
    else:
        raise ValueError(f"No matching model: {args.model}")
    
    if args.model=="MLP":
        X_train_tensor_MH = torch.tensor(X_train_res_MH, dtype=torch.float32).to(device)
        X_train_tensor_WCH = torch.tensor(X_train_res_WCH, dtype=torch.float32).to(device)
        y_train_tensor_MH = torch.tensor(y_train_res_MH, dtype=torch.int64).view(-1, 1).to(device)
        y_train_tensor_WCH = torch.tensor(y_train_res_WCH, dtype=torch.int64).view(-1, 1).to(device)
        optimizer_MH = optim.Adam(model_MH.parameters(), lr=0.001)
        optimizer_WCH = optim.Adam(model_WCH.parameters(), lr=0.001)
        criterion = loss[args.loss]
        for epoch in range(args.epoch):
            outputs_MH = model_MH(X_train_tensor_MH)
            outputs_WCH = model_WCH(X_train_tensor_WCH)
            if args.loss == "BCELoss":
                loss_MH = criterion(outputs_MH[:, 1], y_train_tensor_MH.float()[:, 0])
                loss_WCH = criterion(outputs_WCH[:, 1], y_train_tensor_WCH.float()[:, 0])
            else:
                loss_MH = criterion(outputs_MH, y_train_tensor_MH)
                loss_WCH = criterion(outputs_WCH, y_train_tensor_WCH)
            optimizer_MH.zero_grad()
            optimizer_WCH.zero_grad()
            loss_MH.backward()
            loss_WCH.backward()
            optimizer_MH.step()
            optimizer_WCH.step()
        X_test_tensor = torch.tensor(X_test_red, dtype=torch.float32).to(device)
        X_all_tensor = torch.tensor(X_all_red, dtype=torch.float32).to(device)
        y_pred_MH = model_MH(X_test_tensor).cpu().detach().numpy()[:, 1]
        y_pred_WCH = model_WCH(X_test_tensor).cpu().detach().numpy()[:, 1]
        y_pred_MH_all = model_MH(X_all_tensor).cpu().detach().numpy()[:, 1]
        y_pred_WCH_all = model_WCH(X_all_tensor).cpu().detach().numpy()[:, 1]
        y_prob_MH = y_pred_MH
        y_prob_WCH = y_pred_WCH
        y_prob_MH_all = y_pred_MH_all
        y_prob_WCH_all = y_pred_WCH_all
        
    else:
        model_MH.fit(X_train_res_MH, y_train_res_MH)
        model_WCH.fit(X_train_res_WCH, y_train_res_WCH)
        y_pred_MH = model_MH.predict(X_test_red)
        y_pred_WCH = model_WCH.predict(X_test_red)
        y_pred_MH_all = model_MH.predict(X_all_red)
        y_pred_WCH_all = model_WCH.predict(X_all_red)
        y_prob_MH = 1 - model_MH.predict(X_test_red, return_winning_probability=True)[1]
        y_prob_WCH = 1 - model_WCH.predict(X_test_red, return_winning_probability=True)[1]
        y_prob_MH_all = 1 - model_MH.predict(X_all_red, return_winning_probability=True)[1]
        y_prob_WCH_all = 1 - model_WCH.predict(X_all_red, return_winning_probability=True)[1]
        # MH_model = BPModel(
        #     model=model_MH,
        #     dim_red=dim_red[args.dim_red],
        #     resample=resample[args.resample]
        # )
        # WCH_model = BPModel(
        #     model=model_MH,
        #     dim_red=dim_red[args.dim_red],
        #     resample=resample[args.resample]
        # )
        # MH_model.fit(X_train, y_train)
        # WCH_model.fit(X_train, y_train)
        # MH_model.save("rf_MH.joblib")
        # WCH_model.save("rf_WCH.joblib")
        
    y_pred_MH = np.where(y_pred_MH > 0.5, 1, 0)
    y_pred_WCH = np.where(y_pred_WCH > 0.5, 1, 0)
    acc_MH, ppv_MH, recall_MH, f1_MH = evaluate(y_test, y_pred_MH, label_name+"_MH")
    acc_WCH, ppv_WCH, recall_WCH, f1_WCH = evaluate(y_test, y_pred_WCH, label_name+"_WCH")
    record_MH = {"acc": acc_MH, "ppv": ppv_MH, "recall": recall_MH, "f1": f1_MH}
    record_WCH = {"acc": acc_WCH, "ppv": ppv_WCH, "recall": recall_WCH, "f1": f1_WCH}

    acc_MH, ppv_MH, recall_MH, f1_MH = evaluate(label, y_pred_MH_all, label_name+"_MH")
    acc_WCH, ppv_WCH, recall_WCH, f1_WCH = evaluate(label, y_pred_WCH_all, label_name+"_WCH")
    record_MH_all = {"acc": acc_MH, "ppv": ppv_MH, "recall": recall_MH, "f1": f1_MH}
    record_WCH_all = {"acc": acc_WCH, "ppv": ppv_WCH, "recall": recall_WCH, "f1": f1_WCH}

    # Filter
    # y_test = pd.concat([y_test, y_rest], axis=0)
    # y_rest_np = y_rest.values if isinstance(y_rest, pd.DataFrame) else y_rest.to_numpy()
    # y_pred_MH = np.concatenate([y_pred_MH, y_rest_np[:, 0]], axis=0)
    # y_pred_WCH = np.concatenate([y_pred_WCH, y_rest_np[:, 1]], axis=0)
    # y_prob_MH = np.concatenate([y_prob_MH, y_rest_np[:, 0]], axis=0)
    # y_prob_WCH = np.concatenate([y_prob_WCH, y_rest_np[:, 1]], axis=0)
    return record_MH, record_WCH, record_MH_all, record_WCH_all, (y_test, y_pred_MH, y_pred_WCH, y_prob_MH, y_prob_WCH), (y_pred_MH_all, y_pred_WCH_all, y_prob_MH_all, y_prob_WCH_all)

def repeat_once_benchmark(args, seed, benchmark, label, label_name):
    torch.manual_seed(seed)
    device = torch.device(args.device)
    X_train, X_test, y_train, y_test = train_test_split(benchmark, label, test_size=0.4, random_state=seed)
    
    # Filter
    condition0 = X_test.iloc[:, 0] < 160
    condition1 = X_test.iloc[:, 0] >= 130
    condition2 = X_test.iloc[:, 0] < 130
    condition3 = X_test.iloc[:, 0] >= 120
    condition4 = X_test.iloc[:, 1] < 100
    condition5 = X_test.iloc[:, 1] >= 80
    condition6 = X_test.iloc[:, 1] < 80
    condition_MH = condition2 & condition3 & condition6
    condition_WCH = condition0 & (condition1 & condition5) & condition4
    y_pred_MH = pd.Series(0, index=X_test.index)
    y_pred_WCH = pd.Series(0, index=X_test.index)
    y_pred_MH[condition_MH] = 1
    y_pred_WCH[condition_WCH] = 1

    acc_MH, ppv_MH, recall_MH, f1_MH = evaluate(y_test, y_pred_MH, label_name+"_MH")
    acc_WCH, ppv_WCH, recall_WCH, f1_WCH = evaluate(y_test, y_pred_WCH, label_name+"_WCH")
    record_MH = {"acc": acc_MH, "ppv": ppv_MH, "recall": recall_MH, "f1": f1_MH}
    record_WCH = {"acc": acc_WCH, "ppv": ppv_WCH, "recall": recall_WCH, "f1": f1_WCH}
    return record_MH, record_WCH, (y_test, y_pred_MH, y_pred_WCH)
