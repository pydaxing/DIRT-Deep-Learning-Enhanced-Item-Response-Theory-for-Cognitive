import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):

        q = q.view(-1, q.size()[0], q.size()[1])
        k = k.view(-1, k.size()[0], k.size()[1])
        v = v.view(-1, v.size()[0], v.size()[1])
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
		# 计算softmax
        attention = self.softmax(attention)
		# 添加dropout
        attention = self.dropout(attention)
		# 和V做点积
        context = torch.bmm(attention, v)
        return context.view(context.size()[1], context.size()[2])