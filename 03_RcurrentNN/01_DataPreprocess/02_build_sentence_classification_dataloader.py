from torchtext.vocab import Vocab
import torch
from collections import Counter
import jieba
from torch.utils.data import DataLoader


def tokenizer(s, word=False):
    """
    Args:
        s:
        word: 是否采用分字模式
    Returns:
    """
    if word:
        r = [w for w in s]
    else:
        s = jieba.cut(s, cut_all=False)
        r = " ".join(s).split()
    return r


def build_vocab(tokenizer, filepath, word, min_freq, specials=None):
    """
    根据给定的tokenizer和对应参数返回一个Vocab类
    Args:
        tokenizer:  分词器
        filepath:  文本的路径
        word: 是否采用分字的模式对汉字进行处理
        min_freq: 最小词频，去掉小于min_freq的词
        specials: 特殊的字符，如<pad>，<unk>等
    Returns:
    """
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    counter = Counter()
    with open(filepath, encoding='utf8') as f:
        for string_ in f:
            counter.update(tokenizer(string_.strip(), word))
    return Vocab(counter, min_freq=min_freq, specials=specials)


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
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors


class LoadSentenceClassificationDataset():
    def __init__(self, train_file_path=None,
                 tokenizer=None,
                 batch_size=2,
                 word=True,
                 min_freq=1,
                 max_sen_len='same'):
        """
        Args:
            train_file_path:  训练集路径
            tokenizer:
            batch_size:
            word: 是否采用分字的模式对汉字进行处理
            min_freq: 最小词频，去掉小于min_freq的词
            max_sen_len: 最大句子长度，默认设置其长度为整个数据集中最长样本的长度，即max_sen_len = 'same'
                         当max_sen_len = None时,即在每个batch的所有样本以该batch中最长的样本进行Padding;
        """
        # 根据训练预料建立字典
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.specials = ['<unk>', '<pad>']
        self.word = word
        self.vocab = build_vocab(self.tokenizer,
                                 filepath=train_file_path,
                                 word=self.word,
                                 min_freq=self.min_freq,
                                 specials=self.specials)
        self.PAD_IDX = self.vocab['<pad>']
        self.UNK_IDX = self.vocab['<unk>']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len

    def data_process(self, filepath):
        """
        将每一句话中的每一个词根据字典转换成索引的形式，同时返回所有样本中最长样本的长度
        :param filepath: 数据集路径
        :return:
        [(tensor([61, 36, 58, 45, 33, 37, 40,  4, 39, 18, 16, 23, 12, 49, 35, 30, 51,  3]), tensor(0)),
         (tensor([ 5,  5,  7,  7,  8, 24, 18,  4,  7,  7,  5,  5,  6, 10, 34,  3]), tensor(0)) ...]
        """
        raw_iter = open(filepath, encoding="utf8").readlines()
        data = []
        max_len = 0
        for raw in raw_iter:
            line = raw.rstrip("\n")
            # 问君能有几多愁，恰似一江春水向东流。	0
            # 年年岁岁花相似，岁岁年年人不同。    0
            s, l = line.split('\t')
            tensor_ = torch.tensor([self.vocab[token] for token in
                                    self.tokenizer(s, self.word)], dtype=torch.long)
            l = torch.tensor(int(l), dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_train_val_test_data(self, train_file_paths, val_file_paths, test_file_paths):
        """
        构造Pytorch中的DataLoader
        :param train_file_paths:
        :param val_file_paths:
        :param test_file_paths:
        :return:
        """
        train_data, max_sen_len = self.data_process(train_file_paths)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        val_data, _ = self.data_process(val_file_paths)
        test_data, _ = self.data_process(test_file_paths)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, valid_iter, test_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch中的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以：
        当max_len=None时这就意味着下面的pad_sequence操作，
        最终表现出来的结果就是不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。
        因为此时pad_sequence是以一个batch中最长的样本为标准对其它样本进行padding。

        当max_len = 'same'时，最终表现出来的结果就是，所有样本在padding后的长度都等于训练集中最长样本的长度。
        :param data_batch:
        :return:
        """
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=True,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


if __name__ == '__main__':
    path = 'data_02.txt'
    data_loader = LoadSentenceClassificationDataset(train_file_path=path,
                                                    tokenizer=tokenizer,
                                                    batch_size=5,
                                                    word=True,
                                                    max_sen_len=None)

    train_iter, valid_iter, test_iter = data_loader.load_train_val_test_data(path, path, path)
    for sen, label in train_iter:
        print("batch:", sen)
        print("batch shape:", sen.shape)
        print("labels:", label)
        print("\n")
