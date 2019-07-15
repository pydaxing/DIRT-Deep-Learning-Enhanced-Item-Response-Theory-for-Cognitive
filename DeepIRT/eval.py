from moduel.LSTM import LSTM
from Data.loader import DataLoader
from model.DeepModel import DeepIRT
import tqdm
import numpy as np
from sklearn import metrics
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import random

class myLoss(nn.Module):
    def __init__(self):
        super(myLoss, self).__init__()
    def forward(self, pred, gold):
        loss = 0
        gold = [1 if elem == 1 else 0 for elem in gold]
        for i in range(len(gold)):
            loss += gold[i] * torch.log(pred[i])+ (1 - gold[i]) * torch.log(1 - pred[i])
        return -loss

def performance(p, g):
    p = np.array(p)
    g = np.array(g)
    print(p, g)
    pred = [1 if elem >= 0.5 else 0 for elem in p]
    gold = [1 if elem ==  1 else 0 for elem in g]
    res = [elem for elem in np.abs(np.array(pred) - np.array(gold))]
    acc = 1 - sum(res)/len(res)
    fpr, tpr, thresholds = metrics.roc_curve(g, p)
    auc = metrics.auc(fpr, tpr)
    f1 = metrics.f1_score(gold, pred)
    recall = metrics.recall_score(gold, pred)
    precision = metrics.precision_score(gold, pred)
    mse = metrics.mean_squared_error(g, p)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(gold, pred)
    return acc, auc, f1, recall, precision, mse, rmse, mae

def train_epoch(model, training_data, optimizer, loss_func):
    for batch in tqdm.tqdm(training_data, desc='Training:    ', mininterval=2):
        (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y) = batch
        pred = model(uididx, kcodeidx, qidemb, qidemblength, 'train')
        loss = loss_func(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, optimizer

def test_epoch(model, validation_data, loss_func):
    p = []
    g = []
    for batch in tqdm.tqdm(validation_data, desc='Testing:    ', mininterval=2):
        (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y) = batch
        pred = model(uididx, kcodeidx, qidemb, qidemblength, 'train')
        p += list(pred.view(len(pred)).data.numpy())
        g += list(Y.data.numpy())
    print(performance(p, g))