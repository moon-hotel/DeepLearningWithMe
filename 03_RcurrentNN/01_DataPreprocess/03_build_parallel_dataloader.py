from collections import Counter
from torchtext.vocab import Vocab
import torch
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from torch.utils.data import DataLoader


def my_tokenizer(s):
    s = s.replace(',', " ,").replace(".", " .")
    return s.split()


def build_vocab(tokenizer, filepath, specials=None):
    """
    vocab = Vocab(counter, specials=specials)

    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    # ['<unk>', '<pad>', '<bos>', '<eos>', '.', 'a', 'are', 'A', 'Two', 'in', 'men',...]
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；

    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    # {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3, '.': 4, 'a': 5, 'are': 6,...}
    print(vocab.stoi['are'])  # 通过单词返回得到词表中对应的索引
    """
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    counter = Counter()
    with open(filepath, encoding='utf8') as f:
        for string_ in f:
            counter.update(tokenizer(string_))
    return Vocab(counter, specials=specials)


class LoadEnglishGermanCorpus():
    def __init__(self, train_file_paths=None, tokenizer=None, batch_size=2):
        # 根据训练预料建立英语和德语各自的字典
        self.tokenizer = tokenizer
        self.de_vocab = build_vocab(self.tokenizer, filepath=train_file_paths[0])
        self.en_vocab = build_vocab(self.tokenizer, filepath=train_file_paths[1])
        self.specials = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.PAD_IDX = self.de_vocab['<pad>']
        self.BOS_IDX = self.de_vocab['<bos>']
        self.EOS_IDX = self.de_vocab['<eos>']
        self.batch_size = batch_size

    def data_process(self, filepaths):
        """
        将每一句话中的每一个词根据字典转换成索引的形式
        :param filepaths:
        :return:
        """
        raw_de_iter = iter(open(filepaths[0], encoding="utf8"))
        raw_en_iter = iter(open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_de, raw_en) in zip(raw_de_iter, raw_en_iter):
            de_tensor_ = torch.tensor([self.de_vocab[token] for token in self.tokenizer(raw_de.rstrip("\n"))],
                                      dtype=torch.long)
            en_tensor_ = torch.tensor([self.en_vocab[token] for token in self.tokenizer(raw_en.rstrip("\n"))],
                                      dtype=torch.long)
            data.append((de_tensor_, en_tensor_))

        # [ (tensor([ 9, 37, 46,  5, 42, 36, 11, 16,  7, 33, 24, 45, 13,  4]), tensor([ 8, 45, 11, 13, 28,  6, 34, 31, 30, 16,  4])),
        #   (tensor([22,  5, 40, 25, 30,  6, 12,  4]), tensor([12, 10,  9, 22, 23,  6, 33,  5, 20, 37, 41,  4])),
        #   (tensor([ 8, 38, 23, 39,  7,  6, 26, 29, 19,  4]), tensor([ 7, 27, 21, 18, 24,  5, 44, 35,  4])),
        #   (tensor([ 8, 21,  7, 34, 32, 17, 44, 28, 35, 20, 10, 41,  6, 15,  4]), tensor([ 7, 29,  9,  5, 15, 38, 25, 39, 32,  5, 26, 17,  5, 43,  4])),
        #   (tensor([ 9,  5, 43, 27, 18, 10, 31, 14, 47,  4]), tensor([ 8, 10,  6, 14, 42, 40, 36, 19,  4]))  ]

        return data

    def load_train_val_test_data(self, train_file_paths, val_file_paths, test_file_paths):
        train_data = self.data_process(train_file_paths)
        val_data = self.data_process(val_file_paths)
        test_data = self.data_process(test_file_paths)
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        valid_iter = DataLoader(val_data, batch_size=self.batch_size,
                                shuffle=True, collate_fn=self.generate_batch)
        test_iter = DataLoader(test_data, batch_size=self.batch_size,
                               shuffle=True, collate_fn=self.generate_batch)
        return train_iter, valid_iter, test_iter

    def generate_batch(self, data_batch):
        """
        自定义一个函数来对每个batch的样本进行处理，该函数将作为一个参数传入到类DataLoader中。
        由于在DataLoader中是对每一个batch的数据进行处理，所以这就意味着下面的pad_sequence操作，最终表现出来的结果就是
        不同的样本，padding后在同一个batch中长度是一样的，而在不同的batch之间可能是不一样的。因为pad_sequence是以一个batch中最长的
        样本为标准对其它样本进行padding。 在类encoder-decoder框架中，一般只需要保证同batch中样本长度相同即可
        :param data_batch:
        :return:
        """
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            de_batch.append(de_item)  # 编码器输入序列不需要加起止符
            # 在每个idx序列的首位加上 起始token 和 结束 token
            en = torch.cat([torch.tensor([self.BOS_IDX]), en_item, torch.tensor([self.EOS_IDX])], dim=0)
            en_batch.append(en)
        # 以最长的序列为标准进行填充
        de_batch = pad_sequence(de_batch, padding_value=self.PAD_IDX)  # [de_len,batch_size]
        en_batch = pad_sequence(en_batch, padding_value=self.PAD_IDX)  # [en_len,batch_size]
        return de_batch, en_batch


if __name__ == '__main__':
    train_filepath = ['train_.de',
                      'train_.en']

    data_loader = LoadEnglishGermanCorpus(train_filepath, tokenizer=my_tokenizer, batch_size=2)
    train_iter, valid_iter, test_iter = data_loader.load_train_val_test_data(train_filepath,
                                                                             train_filepath,
                                                                             train_filepath)
    print(len(train_iter))
    for src, tgt in train_iter:
        print("src shape：", src.shape)  # [de_tensor_len,batch_size]
        print("tgt shape:", tgt.shape)  # [de_tensor_len,batch_size]
        print("===================》")
