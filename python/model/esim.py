import torch
import torch.nn as nn
from transformer.attention import attention


class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, TEXT, batch_size):
        super(ESIM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # nn.init.xavier_uniform_(self.embedding.weight) # 初始化权重
        self.embedding.weight.data.copy_(TEXT.vocab.vectors) # 载入预训练词向量
        
        self.LSTM_stack1 = nn.LSTM(embedding_dim, hidden_size=batch_size, num_layers=2, batch_first=True, bidirectional=True)
        self.LSTM_stack2 = nn.LSTM(embedding_dim, hidden_size=batch_size, num_layers=2, batch_first=True, bidirectional=True)

        self.LSTM_stack_2 = nn.LSTM(200, 64, num_layers=2, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(512, 1024)
        # self.norm1 = torch.nn.BatchNorm1d(1024, momentum=0.5)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x1, x2):

        x1 = x1.permute(1, 0)
        x2 = x2.permute(1, 0)

        x1 = self.embedding(x1)
        x2 = self.embedding(x2)

        batch_num = x1.size(0)

        # 对两个输入序列用BiLSTM分析上下文含义，重新进行embedding
        x1, _ = self.LSTM_stack1(x1.float())  # (batch, sentence_len, hidden_units)

        x2, _ = self.LSTM_stack2(x2.float())
        
        # 整个矩阵乘同一个数可以保留词向量之间的关系并放大它们之间的差异
        norm_const_x1 = torch.abs(torch.reciprocal_(torch.mean(x1)))
        norm_const_x2 = torch.abs(torch.reciprocal_(torch.mean(x2)))
        x1_norm = x1 * norm_const_x1
        x2_norm = x2 * norm_const_x2

        output = torch.tensor([])

        for k in range(batch_num):
            # 为了方便理解，这里对batch里面的每个句子分别计算注意力，并把结果合并在一起
            # 但其实这样计算效率更慢，直接用矩阵计算更好

            # 注意力机制
            x1_tilde, _ = attention(x1_norm[k], x2_norm[k], x2_norm[k])
            x2_tilde, _ = attention(x2_norm[k], x1_norm[k], x1_norm[k])

            # Enhancement of local inference information
            # 计算句子间的差异信息

            x1_diff = torch.sub(x1_norm[k], x1_tilde)  
            x2_diff = torch.sub(x2_norm[k], x2_tilde)  

            x1_mul = torch.mul(x1_norm[k], x1_tilde)  
            x2_mul = torch.mul(x2_norm[k], x2_tilde)

            m_x1 = torch.cat([x1_norm[k], x1_tilde, x1_diff, x1_mul], 0)  
            m_x2 = torch.cat([x2_norm[k], x2_tilde, x2_diff, x2_mul], 0)  

            m_x1 = torch.unsqueeze(m_x1, 0)  
            m_x2 = torch.unsqueeze(m_x2, 0)  

            # Inference Composition
            # 用BiLSTM分析overall inference relationship between x1 and x2

            v1_outs, _ = self.LSTM_stack_2(m_x1.float())  # (batch, sentence_len, hidden_units)
            v2_outs, _ = self.LSTM_stack_2(m_x2.float())

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
            # v_fc1 = torch.relu(self.drop1(v_fc1))  
            v_fc1 = torch.relu(v_fc1)  
            v_fc2 = self.fc2(v_fc1)
            v_fc2 = torch.relu(v_fc2)  
            v_fc3 = self.fc3(v_fc2)

            result = v_fc3 / torch.norm(v_fc3)
            result = result.unsqueeze(0)

            output = torch.cat([output, result], 0)  

        # print('output: ', output)
        return output
