import torch.nn as nn
import copy
from transformer.attention import attention
import torch


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.p = dropout
        self.linears = clones(nn.Linear(300, 300), 4)
        self.attn = None

    def forward(self, query, key, value):

        batche_size = query.size(0)
        seq_len = query.size(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # 对qkv做多次linears变换（相当于原论文中乘矩阵），并将结果拼接在一起（通过view函数）
        # 这里就是缩写，其实就是对q、k、v分别应用线性变换，并把结果矩阵调整为输入的尺寸
        query, key, value = [l(x).view(batche_size, self.h*self.d_k, seq_len).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, dropout=self.p)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batche_size, -1, self.h * self.d_k)
        return self.linears[-1](x)
