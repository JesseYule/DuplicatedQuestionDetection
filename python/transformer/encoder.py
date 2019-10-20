# Standard PyTorch imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable
import matplotlib.pyplot as plt
from transformer.multiHeadAttention import MultiHeadedAttention
from transformer.positionwiseFeedForward import PositionwiseFeedForward


def clones(module, N):
    "Produce N identical layers."
    # 对同一个module复制N次
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input (and mask) through each layer in turn."
        # 相当于把x输入到N个堆叠的layers中计算输出
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# encoderlayer是encoder的一个重要构成部分
class EncoderLayer(nn.Module):
    "Encoder is made up of two sublayers, self-attn and feed forward (defined below)"
    def __init__(self, size, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        self.self_attn = MultiHeadedAttention(6, 300)

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        # 一共复制了两个sublayer函数
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)


# layernorm主要对encoderlayer做标准化处理
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # 所以最后一维一定要是词向量维度
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # eps防止分母为0，b2防止梯度消失，整体是标准化处理
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# encoderlayer中的残差连接
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity we apply the norm first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer function that maintains the same size."
        return x + self.dropout(sublayer(self.norm(x)))

