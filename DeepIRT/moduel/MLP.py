import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size = 1, dropout = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.output = nn.Sequential(
                                    nn.Linear(self.input_size, self.hidden_size),
                                    nn.Linear(self.hidden_size, self.output_size))

    def forward(self, X):
        outputs = self.output(X)
        return outputs

# mlp = DNN(5, 3, 1)
# data = torch.FloatTensor([[2,3,4,5,6],[1,2,3,4,5]])
# print(mlp(data))