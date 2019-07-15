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
import eval
lstm = LSTM(50, 5)

def preparing():
    path = '/home/chengsong/experimentation/Project/DeepIRT/dataset/'
    train_uid_qid_res = pickle.load(open(path + 'train_uid_qid_res.pkl', 'rb'))
    test_uid_qid_res = pickle.load(open(path + 'test_uid_qid_res.pkl', 'rb'))
    qid_emb = pickle.load(open(path + 'qid_emb_data.pkl', 'rb'))
    qid_kcode = pickle.load(open(path + 'qid_kcode_data.pkl', 'rb'))
    kcode_emb = pickle.load(open(path + 'code_emb_data.pkl', 'rb'))
    trainLoader, testLoader = DataLoader(train_uid_qid_res, test_uid_qid_res, qid_emb, kcode_emb, qid_kcode)
    return trainLoader, testLoader

if __name__ == '__main__':
    train, test = preparing()
    model = DeepIRT(50, 1, 50, 20, 1, 2000, 20000, 50)  # example : 50, 1, 50, 20, 1, 2000, 20000, 50,
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = eval.myLoss()
    for epoch in range(1000):
        model, optimizer = eval.train_epoch(model,train,optimizer,loss_func)
        eval.test_epoch(model,test,loss_func)



