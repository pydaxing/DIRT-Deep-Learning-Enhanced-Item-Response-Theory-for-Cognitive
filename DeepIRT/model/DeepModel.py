import torch
import torch.nn as nn
from torch.autograd import Variable
from moduel.LSTM import LSTM
from moduel.Embedding import DenseEmbedding
from moduel.MLP import DNN
from moduel.Attention import ScaledDotProductAttention
from moduel.IRT import IRT

import numpy as np

class DeepIRT(nn.Module):
    def __init__(self, lstm_input_size=None, lstm_hidden_size=None, dnn_input_size=None, dnn_hidden_size=None,
                 dnn_output_size=None,knowledge_nums=None, student_nums=None, denseDim=None):
        super().__init__()
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dnn_input_size = dnn_input_size
        self.dnn_hidden_size = dnn_hidden_size
        self.dnn_output_size = dnn_output_size
        self.knowledge_nums = knowledge_nums
        self.denseDim = denseDim
        self.student_nums = student_nums

        self.A = DNN(self.dnn_input_size, self.dnn_hidden_size, self.dnn_output_size)
        self.Theta = DNN(self.dnn_input_size, self.dnn_hidden_size, self.dnn_output_size)
        self.B = LSTM(self.lstm_input_size, self.lstm_hidden_size)
        self.studentEmbed = nn.Embedding(self.student_nums, self.denseDim)
        self.knowledgeEmbed = nn.Embedding(self.knowledge_nums, self.denseDim)
        self.atte = ScaledDotProductAttention()
        self.irt = IRT()

    def forward(self, uididx, kcodeidx, qidemb, qidemblength, mode):
        self.stuEmb = self.studentEmbed(Variable(uididx))
        self.kcodeemb = [self.knowledgeEmbed(Variable(elem)) for elem in kcodeidx]
        self.mastvec = []
        self.avec = []
        self.bvec = []
        for i in range(len(self.stuEmb)):
            mastery = torch.sigmoid(torch.mm(self.kcodeemb[i], self.stuEmb[i].view(len(self.stuEmb[i]), -1))/5.0)
            self.mastvec.append(torch.sum(torch.mul(mastery, self.kcodeemb[i]), 0))
            self.avec.append(torch.sum(self.kcodeemb[i], 0))
            self.bvec.append(self.atte(qidemb[i], self.kcodeemb[i], self.kcodeemb[i], scale = 0.15))
            if mode == 'test':
                print(mastery)

        self.mastvec = torch.stack(self.mastvec)
        self.avec = torch.stack(self.avec)
        self.bvec = torch.stack(self.bvec)

        t = self.Theta(self.mastvec)
        a = 8 * (torch.sigmoid(torch.abs(self.A(self.avec)))-0.5)
        b = 8 * (torch.sigmoid(self.B(self.bvec, qidemblength)) - 0.5)

        if mode == 'test':
            print(t,a, b)
        return self.irt(t, a, b)