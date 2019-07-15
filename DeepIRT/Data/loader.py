import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.utils.data as Data
from torchvision import transforms
import pandas as pd
import numpy as np
from functools import reduce
from Constant import Constant as C
import pickle

uid_index_dict = {}
qid_index_dict = {}
kcode_index_dict = {}

def id2index(ids):
    idlist = list(set(ids))
    id_index = {k: v for (k, v) in zip(idlist, [i for i in range(len(idlist))])}
    return id_index

def do_id2index(train, test,qid_kcode):
    global uid_index_dict, qid_index_dict, kcode_index_dict
    uid_index_dict = id2index(set(train['userId']).union(set(test['userId'])))
    qid_index_dict = id2index(set(qid_kcode.keys()))
    kcode_index_dict = id2index(reduce(lambda x, y: list(set(x).union(set(y))), qid_kcode.values()))

class DeepirtDataset(Dataset):
    # uid_qid_res dataframe # qid_emb dict # kcode_emb dataframe # qid_kcode dict
    def __init__(self, uid_qid_res, qid_emb, kcode_emb, qid_kcode):
        self.uid_qid_res = uid_qid_res
        self.qid_emb = qid_emb
        self.kcode_emb = kcode_emb
        self.qid_kcode = qid_kcode

    def __len__(self):
        return len(self.uid_qid_res)

    def __getitem__(self, index):
        if type(index) == slice:
            lines = self.uid_qid_res[index]
        else:
            lines = self.uid_qid_res[index:index+1]
        uididx = torch.LongTensor([uid_index_dict[uid] for uid in lines['userId']])
        qididx = torch.LongTensor([qid_index_dict[tid] for tid in lines['topicId']])
        Y = torch.FloatTensor((lines['result']%2).astype(int).tolist())
        kcodeidx = [list(map(lambda x : torch.LongTensor([kcode_index_dict[x]]), self.qid_kcode[qid])) for qid in lines['topicId']]
        ### padding and pack
        qidemblength = np.array([len(self.qid_emb[tid]) for tid in lines['topicId']])
        qidemb = np.array([torch.nn.functional.pad(torch.FloatTensor(self.qid_emb[tid]), (0, 0, 0, 30-torch.FloatTensor(self.qid_emb[tid]).size()[0]), "constant", 0).data.numpy() for tid in lines['topicId']])
        qidemb = torch.FloatTensor(qidemb)

        kcodeemb = []
        for tid in lines['topicId']:
            embedding = []
            for kcode in self.qid_kcode[tid]:
                embedding.append(self.kcode_emb[self.kcode_emb[0] == kcode].values[:,1:][0])
            kcodeemb.append(torch.FloatTensor(np.array(embedding).astype(float).tolist()))
        return (uididx, kcodeidx, qidemb, qidemblength, kcodeemb, Y)

def collate_fn(batch):
    uididx = []
    kcodeidx = []
    qidemblength = []
    kcodeemb = []
    Y = []
    for elem in batch:
        uididx.append(elem[0])
        kcodeidx.append(elem[1][0])
        qidemblength.append(elem[3][0])
        kcodeemb.append(elem[4][0])
        Y.append(elem[5])
    qidemb = torch.stack([elem[2][0] for elem in batch], 0)
    return torch.LongTensor(uididx), [torch.LongTensor(elem) for elem in kcodeidx], qidemb, torch.LongTensor(np.array(qidemblength)), kcodeemb, torch.stack(Y).view(-1).long()

def DataLoader(train, test, qid_emb, kcode_emb, qid_kcode):
    do_id2index(train, test, qid_kcode)
    trainDataset = DeepirtDataset(train, qid_emb, kcode_emb, qid_kcode)
    testDataset = DeepirtDataset(test, qid_emb, kcode_emb, qid_kcode)
    trainLoader = Data.DataLoader(trainDataset, batch_size= C.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    testLoader = Data.DataLoader(testDataset, batch_size= C.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    return trainLoader, testLoader

