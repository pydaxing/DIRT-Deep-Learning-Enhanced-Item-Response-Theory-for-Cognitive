import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class DenseEmbedding(nn.Module):
    def __init__(self, numbers, output_size):
        super().__init__()
        self.numbers = numbers
        self.output_size = output_size
        self.embedding = nn.Embedding(self.numbers, output_size)
    def forward(self, X):
        denseEmbedding = self.embedding(X)
        return denseEmbedding


# dense = DenseEmbedding(5, 3)
#
# data = torch.LongTensor([1,2,3,4])
# print(dense(data))