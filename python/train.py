from data_preprocess import data_iter
from model.esim_transformer import ESIMTransformer
import torch.optim as optim
import torch.nn as nn
import os
import torch

datapath, trainset_name, validset_name, filetype = '../data', 'testtrain.csv',\
                                                   'testvalid.csv', 'csv'


train_iter, val_iter, TEXT, batchsize = data_iter(datapath, trainset_name, validset_name, filetype)

esimtransformer = ESIMTransformer(len(TEXT.vocab), 300, 32, 1, TEXT, batchsize)

optimizer = optim.Adam(esimtransformer.parameters(), lr=1e-2)

loss_fn = nn.CrossEntropyLoss()

# if os.path.exists('model_checkpoint/model.pkl'):
#     print('load model')
#     esimtransformer.load_state_dict(torch.load('model_checkpoint/model.pkl'))

for epoch in range(100):
    train_acc = 0
    train_loss = 0
    min_loss = 1e5
    for i, batch in enumerate(train_iter):

        optimizer.zero_grad()

        x1 = batch.question1
        x2 = batch.question2
        y = batch.is_duplicate

        preds = esimtransformer(x1, x2)

        train_loss = loss_fn(preds, y)

        train_loss += train_loss.item()

        print('train loss: ', train_loss)

        # 验证集
        val_acc = 0
        val_loss = 0
        val_batch = next(iter(val_iter))
        val_preds = esimtransformer(val_batch.question1, val_batch.question2)
        _, result = torch.max(val_preds, 1)
        correct = 0
        for k in range(result.size()[0]):
            if result[k] - val_batch.is_duplicate[k] == 0:
                correct += 1
        print('correct amount: ', correct)
        print('valid accuracy： ', correct / result.size()[0])

        # if train_loss < min_loss:
        #     print('save model')
        #     torch.save(esimtransformer.state_dict(), 'model_checkpoint/model.pkl')
        #     min_loss = train_loss

        train_loss.backward()
        optimizer.step()
