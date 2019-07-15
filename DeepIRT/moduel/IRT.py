import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class IRT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, theta, a, b):
        exp = torch.exp(-1.7*a*(theta - b))
        outputs = 1 / (1 + exp)
        # print(outputs)
        return outputs
