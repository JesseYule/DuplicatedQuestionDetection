from transformer.positionalEncoding import get_sinusoid_encoding_table
from transformer.encoder import *
from transformer.positionwiseFeedForward import PositionwiseFeedForward

import torch
import torch.nn as nn


class DoubleEncoder(torch.nn.Module):

    def __init__(self, vector_size=300, heads=8, hidden_layers=200, dropout=0.1):
        super(DoubleEncoder, self).__init__()
        self.feedforward = PositionwiseFeedForward(vector_size, hidden_layers)
        self.layer1 = EncoderLayer(vector_size, self.feedforward, dropout)
        self.layer2 = EncoderLayer(vector_size, self.feedforward, dropout)
        self.encoder1 = Encoder(self.layer1, heads)
        self.encoder2 = Encoder(self.layer2, heads)

    def forward(self, x1, x2, batch_size=100):

        # 计算positional encoding
        x1_n_position = x1.size()[1]
        x1_d_hid = x1.size()[2]
        x1_sinusoid_table = get_sinusoid_encoding_table(x1_n_position, x1_d_hid)

        x2_n_position = x2.size()[1]
        x2_d_hid = x2.size()[2]
        x2_sinusoid_table = get_sinusoid_encoding_table(x2_n_position, x2_d_hid)

        x1_sinusoid_table = x1_sinusoid_table.repeat(batch_size, 1, 1)
        x2_sinusoid_table = x2_sinusoid_table.repeat(batch_size, 1, 1)

        # 在原始向量加入位置编码
        x1_p = x1 + x1_sinusoid_table
        x2_p = x2 + x2_sinusoid_table

        x1_output = self.encoder1(x1_p)
        x2_output = self.encoder2(x2_p)

        return x1_output, x2_output
