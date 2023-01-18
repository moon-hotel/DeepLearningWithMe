from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
import torch
import re
from tqdm import tqdm


def my_tokenizer(s):
    tokenizer = get_tokenizer('basic_english')
    return tokenizer(s)


def clean_str(string):
    string = re.sub("[^A-Za-z0-9\-\?\!\.\,]", " ", string).lower()
    return string


def build_vocab(tokenizer, filepath, min_freq, specials=None):
    """
    根据给定的tokenizer和对应参数返回一个Vocab类
    Args:
        tokenizer:  分词器
        filepath:  文本的路径
        min_freq: 最小词频，去掉小于min_freq的词
        specials: 特殊的字符，如<pad>，<unk>等
    Returns:
    """
    if specials is None:
        specials = ['<unk>', '<pad>']
    counter = Counter()
    with open(filepath, encoding='utf8') as f:
        for string_ in tqdm(f):
            string_ = string_.strip().split('","')[-1][:-1]  # 新闻描述
            counter.update(tokenizer(clean_str(string_)))
    return Vocab(counter, min_freq=min_freq, specials=specials)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len : 最大句子长度，默认为None，即在每个batch中以最长样本的长度对其它样本进行padding；
        当指定max_len的长度小于一个batch中某个样本的长度，那么在这个batch中还是会以最长样本的长度对其它样本进行padding
        建议指定max_len的值为整个数据集中最长样本的长度
    Returns:
    """
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    length = max_len
    max_len = max([s.size(0) for s in sequences])
    if length is not None:
        max_len = max(length, max_len)
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor


class LoadSentenceClassificationDataset():
    def __init__(self, train_file_path=None,  # 训练集路径
                 tokenizer=None,
                 batch_size=20,
                 min_freq=1,  # 最小词频，去掉小于min_freq的词
                 max_sen_len='same'):  # 最大句子长度，默认设置其长度为整个数据集中最长样本的长度
        # max_sen_len = None时，表示按每个batch中最长的样本长度进行padding
        # 根据训练预料建立字典
        self.tokenizer = tokenizer
        self.min_freq = min_freq
        self.specials = ['<unk>', '<pad>']
        self.vocab = build_vocab(self.tokenizer,
                                 filepath=train_file_path,
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
        """

        raw_iter = open(filepath,encoding='utf8').readlines()
        data = []
        max_len = 0
        for raw in tqdm(raw_iter, ncols=80):
            line = raw.rstrip("\n").split('","')
            s, l = line[-1][:-1], line[0][1:]
            s = clean_str(s)
            tensor_ = torch.tensor([self.vocab[token] for token in
                                    self.tokenizer(s)], dtype=torch.long)
            l = torch.tensor(int(l) - 1, dtype=torch.long)
            max_len = max(max_len, tensor_.size(0))
            data.append((tensor_, l))
        return data, max_len

    def load_train_val_test_data(self, train_file_paths, test_file_paths):
        train_data, max_sen_len = self.data_process(train_file_paths)  # 得到处理好的所有样本
        if self.max_sen_len == 'same':
            self.max_sen_len = max_sen_len
        test_data, _ = self.data_process(test_file_paths)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, test_iter

    def generate_batch(self, data_batch):
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                      padding_value=self.PAD_IDX,
                                      batch_first=False,
                                      max_len=self.max_sen_len)
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


if __name__ == '__main__':
    path = "./data/ag_news_csv/test.csv"
    data_loader = LoadSentenceClassificationDataset(train_file_path=path,
                                                    tokenizer=my_tokenizer,
                                                    max_sen_len=None)
    data, max_len = data_loader.data_process(path)
    train_iter, test_iter = data_loader.load_train_val_test_data(path, path)
    for sample, label in train_iter:
        print(sample.shape)  # [seq_len,batch_size]
