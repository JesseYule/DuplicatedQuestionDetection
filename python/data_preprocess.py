from torchtext.data import Field, TabularDataset
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
import os
import torch
from torchtext import data


# 处理未知单词
def init_emb(vocab, init="randn", num_special_toks=2):
    emb_vectors = vocab.vectors
    sweep_range = len(vocab)
    running_norm = 0.
    num_non_zero = 0
    total_words = 0
    for i in range(num_special_toks, sweep_range):
        if len(emb_vectors[i, :].nonzero()) == 0:
            # std = 0.05 is based on the norm of average GloVE 100-dim word vectors
            if init == "randn":
                torch.nn.init.normal(emb_vectors[i], mean=0, std=0.05)
        else:
            num_non_zero += 1
            running_norm += torch.norm(emb_vectors[i])
        total_words += 1
    print("average GloVE norm is {}, number of known words are {}, total number of words are {}".format(
        running_norm / num_non_zero, num_non_zero, total_words))


def data_iter(datapath, trainset_name, validset_name, filetype):

    # field对象指定怎么处理数据，token表示将str变为token
    # sequential表示是否切分数据，如果数据已经序列化且是数字类型，use_vocab就是False，这里label本身是数字，所以直接false
    tokenize = lambda x: x.split()
    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    LABEL = Field(sequential=False, use_vocab=False)

    # 之前在field对象中说明了怎么处理不同的数据，现在具体数名各个列的数据具体属于哪个field，none的就是不用处理的数据
    train_fields = [("id", None),  # 不需要这些列，所以为none
                    ("qid1", None),
                    ("qid2", None),
                    ("question1", TEXT),
                    ("question2", TEXT),
                    ("is_duplicate", LABEL)]

    # 正式读数据，构建Dataset对象
    trn, vld = TabularDataset.splits(
        path=datapath,  # 数据存放的根目录
        train=trainset_name, validation=validset_name,
        format=filetype,
        skip_header=True,  # 如果你的csv有表头, 确保这个表头不会作为数据处理
        fields=train_fields)

    # 构建词表
    # 先统计训练集里面一共出现了什么单词，添加到词汇表中
    # 建立单词在词汇表位置的映射，把句子的单词替换为数字
    # 词汇表也会从预训练的embedding模型（glove）读取需要的词向量
    # 所以最终模型通过句子单词在词汇表的位置、词汇表与glove的词向量一一对应起来

    cache = 'mycache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='../../../programming/glove/glove.42B.300d.txt', cache=cache)
    # vectors.unk_init = init.xavier_uniform_
    TEXT.build_vocab(trn, vectors=vectors)

    batchsize = 100
    # 构建迭代器，方便训练模型的批处理
    train_iter, val_iter = BucketIterator.splits(
        (trn, vld),
        # 我们把Iterator希望抽取的Dataset传递进去
        batch_sizes=(batchsize, batchsize),  # 这里指train和valid分别的batch size
        device=None,
        # 如果要用GPU，这里指定GPU的编号
        sort_key=lambda x: len(x.question1),
        # BucketIterator 依据什么对数据分组
        sort_within_batch=False,
        repeat=False)

    return train_iter, val_iter, TEXT, batchsize




