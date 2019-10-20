import torch
import torch.nn as nn

from transformer.attention import attention
from transformer.doubleEncoder import DoubleEncoder


class ESIMTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, TEXT, batch_size):
        super(ESIMTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # nn.init.xavier_uniform_(self.embedding.weight) # 初始化权重
        self.embedding.weight.data.copy_(TEXT.vocab.vectors)  # 载入预训练词向量

        self.linear_transform = nn.Linear(300, 512)

        self.fc1 = nn.Linear(1200, 1200)
        self.fc2 = nn.Linear(1200, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x1, x2):

        x1 = x1.permute(1, 0)
        x2 = x2.permute(1, 0)

        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        # transformer
        double_encoder1 = DoubleEncoder(300, 8, 200, 0.1)
        x1_norm, x2_norm = double_encoder1(x1, x2)

        # ESIM
        x1_tilde, _ = attention(x1_norm, x2_norm, x2_norm)
        x2_tilde, _ = attention(x2_norm, x1_norm, x1_norm)

        # Enhancement of local inference information
        # 计算句子间的差异信息
        x1_diff = torch.sub(x1_norm, x1_tilde)
        x2_diff = torch.sub(x2_norm, x2_tilde)

        x1_mul = torch.mul(x1_norm, x1_tilde)
        x2_mul = torch.mul(x2_norm, x2_tilde)

        m_x1 = torch.cat([x1_norm, x1_tilde, x1_diff, x1_mul], 1)
        m_x2 = torch.cat([x2_norm, x2_tilde, x2_diff, x2_mul], 1)

        # Inference Composition
        double_encoder2 = DoubleEncoder(300, 8, 200, 0.1)
        v1_outs, v2_outs = double_encoder2(m_x1, m_x2, m_x1.size(0))

        # 整个矩阵乘同一个数可以保留词向量之间的关系并放大它们之间的差异
        norm_const_v1 = torch.abs(torch.reciprocal_(torch.mean(v1_outs)))
        norm_const_v2 = torch.abs(torch.reciprocal_(torch.mean(v1_outs)))
        v1_outs_norm = v1_outs * norm_const_v1
        v2_outs_norm = v2_outs * norm_const_v2

        # Pooling Layer
        v_1_sum = torch.sum(v1_outs_norm, 1)
        v_1_ave = torch.div(v_1_sum, v_1_sum.size()[1])

        v_2_sum = torch.sum(v2_outs_norm, 1)
        v_2_ave = torch.div(v_2_sum, v_2_sum.size()[1])

        v_1_max = torch.max(v1_outs_norm, 1)[0]
        v_2_max = torch.max(v2_outs_norm, 1)[0]

        v = torch.cat([v_1_ave, v_2_ave, v_1_max, v_2_max], 1).squeeze()

        v_fc1 = self.fc1(v)
        v_fc1 = torch.relu(v_fc1)
        v_fc2 = self.fc2(v_fc1)
        v_fc2 = torch.relu(v_fc2)
        v_fc3 = self.fc3(v_fc2)

        result = v_fc3 / torch.norm(v_fc3)

        return result
