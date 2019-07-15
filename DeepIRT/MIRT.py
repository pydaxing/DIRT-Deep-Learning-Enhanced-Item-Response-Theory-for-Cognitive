from moduel.LSTM import LSTM
from Data.loader import DataLoader
from model.DeepModel import DeepMIRT, DeepIRT
import tqdm
import numpy as np
from sklearn import metrics
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import eval

def performance(p, g):
    p = np.array(p)
    g = np.array(g)
    # pred = [1 if elem < 0.7 else 0 for elem in p]
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

class MIRT(nn.Module):
    def __init__(self,  denseDim=500):
        super().__init__()
        self.Theta = nn.Embedding(num_embeddings=20000, embedding_dim=denseDim)

        self.A = nn.Linear(50, denseDim)

        self.B = nn.Linear(50, 1)

    def forward(self, students, questions):
        thetas = self.Theta(students)
        a = self.A(torch.sigmoid(torch.sum(questions, 1)))
        b = self.B(torch.sigmoid(torch.sum(questions, 1)))
        e = b.view(-1) + torch.sum(torch.mul(thetas, a), 1)
        res = torch.sigmoid(torch.exp(e) / (1 + torch.exp(e)))

        return res.view(len(res), -1)





def preparing():
    path = '/home/chengsong/experimentation/Project/DeepIRT/dataset/'
    train_uid_qid_res = pickle.load(open(path + 'train_uid_qid_res.pkl', 'rb'))[0:1000]
    test_uid_qid_res = pickle.load(open(path + 'test_uid_qid_res.pkl', 'rb'))[2000:2200]
    # test_uid_qid_res = test_uid_qid_res[test_uid_qid_res['result'] == 0]
    print(sum(test_uid_qid_res['result'])/len(test_uid_qid_res))
    qid_emb = pickle.load(open(path + 'qid_emb_data.pkl', 'rb'))
    qid_kcode = pickle.load(open(path + 'qid_kcode_data.pkl', 'rb'))
    kcode_emb = pickle.load(open(path + 'code_emb_data.pkl', 'rb'))
    trainLoader, testLoader = DataLoader(train_uid_qid_res, test_uid_qid_res, qid_emb, kcode_emb, qid_kcode)
    return trainLoader, testLoader


if __name__ == '__main__':
    train, test = preparing()

    model = MIRT(50)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_func = eval.mseLoss()
    for epoch in range(1000):
        for batch in tqdm.tqdm(train, desc='Training:    ', mininterval=2):
            (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y) = batch
            pred = model(uididx, qidemb)
            loss = loss_func(pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        p = []
        g = []
        for batch in tqdm.tqdm(test, desc='Testing:    ', mininterval=2):
            (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y) = batch
            pred = model(uididx, qidemb)

            p += list(pred.view(len(pred)).data.numpy())
            g += list(Y.data.numpy())

        print(performance(p, g))