"""
文件名: Code/utils/data_helper.py
创建时间: 2023/5/6 8:07 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import os
from collections import Counter
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_HOME = os.path.join(PROJECT_HOME, 'data')


class Vocab(object):
    """
    构建词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[num])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['[UNK]'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度

    :param top_k:  取出现频率最高的前top_k个token
    :return:
    以下为以字为粒度构建的词表
    [['上', '联', '：', '一', '夜', '春', '风', '去', '，', '怎', '么', '对', '下', '联', '？'],
     ['隋', '唐', '五', '代', '｜', '欧', '阳', '询', '《', '温', '彦', '博', '碑', '》', '，', '书',...]]




    为词为粒度构建的词表与上面类似
    """
    UNK = '[UNK]'  # 0
    PAD = '[PAD]'  # 1

    def __init__(self, top_k=2000, data=None):
        counter = Counter()
        self.stoi = {Vocab.UNK: 0, Vocab.PAD: 1}
        self.itos = [Vocab.UNK, Vocab.PAD]
        for text in data:
            token = tokenize(text)
            # ['上', '联', '：', '一', '夜', '春', '风', '去', '，', '怎', '么', '对', '下', '联', '？']
            counter.update(token)  # 统计每个token出现的频率
        top_k_words = counter.most_common(top_k)
        for i, word in enumerate(top_k_words):
            self.stoi[word[0]] = i + 2  # 2表示已有了UNK和PAD
            self.itos.append(word[0])

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def tokenize(text):
    words = " ".join(text).split()
    return words


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            padding_content = [padding_value] * (max_len - tensor.size(0))
            tensor = torch.cat([tensor, torch.tensor(padding_content)], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class TouTiaoNews(object):
    """
    头条新闻标题数据集，一共15个类别:
    ['故事','文化','娱乐','体育','财经','房产','汽车','教育','科技','军事','旅游','国际','股票','三农','游戏']
    训练集:验证集:测试集
    """

    def __init__(self, top_k=2000,
                 max_sen_len=None,
                 batch_size=4,
                 is_sample_shuffle=True):
        self.data_path = os.path.join(DATA_HOME, 'toutiao')
        self.top_k = top_k
        self.data_path_train = os.path.join(self.data_path, 'toutiao_train.txt')
        # self.data_path_train = os.path.join(self.data_path, 'test.txt')
        self.data_path_val = os.path.join(self.data_path, 'toutiao_val.txt')
        self.data_path_test = os.path.join(self.data_path, 'toutiao_test.txt')
        raw_data_train, _ = self.load_raw_data(self.data_path_train)
        self.vocab = Vocab(top_k=self.top_k, data=raw_data_train)
        self.max_sen_len = max_sen_len
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    @staticmethod
    def load_raw_data(file_path=None):
        """
        载入原始的文本
        :param file_path:
        :return:
        samples: ['上联：一夜春风去，怎么对下联？', '隋唐五代｜欧阳询《温彦博碑》，书于贞观十一年，是欧最晚的作品']
        labels: ['1','1']
        """
        samples, labels = [], []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('_!_')
                samples.append(line[0])
                labels.append(line[1])
        return samples, labels

    def data_process(self, file_path):
        samples, labels = self.load_raw_data(file_path)
        print(samples[:10])
        data = []
        for i in tqdm(range(len(samples)), ncols=80):
            tokens = tokenize(samples[i])
            token_ids = [self.vocab[token] for token in tokens]
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            l = torch.tensor(int(labels[i]), dtype=torch.long)
            data.append((token_ids_tensor, l))
        return data

    def generate_batch(self, data_batch):

        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.vocab.stoi[self.vocab.PAD],
                                      batch_first=True,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label

    def load_train_val_test_data(self, is_train=False):
        test_data = self.data_process(self.data_path_test)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=False, collate_fn=self.generate_batch)
        if not is_train:
            return test_iter
        train_data = self.data_process(self.data_path_train)  # 得到处理好的所有样本
        val_data = self.data_process(self.data_path_val)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        return train_iter, test_iter, val_iter
