import torch
import torch.nn.functional as F
import math


def attention(query, key, value, dropout=0.0):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(1)  # 词向量长度，较大的dk会导致内积结果较大，导致softmax非0即1
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 归一化处理
    p_attn = F.softmax(scores, dim=-1)

    # (Dropout described below)
    p_attn = F.dropout(p_attn, p=dropout)

    return torch.matmul(p_attn, value), p_attn
