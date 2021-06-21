from torchtext.vocab import Vocab
from collections import Counter
import jieba


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


def my_tokenizer(s):
    """
    返回tokenize后的结果
    """
    s = s.replace(',', " ,").replace(".", " .")
    return s.split()


def build_vocab(tokenizer, filepath, word, min_freq, specials=None):
    if specials is None:
        specials = ['<unk>', '<pad>', '<bos>', '<eos>']
    counter = Counter()
    with open(filepath, encoding='utf8') as f:
        for string_ in f:
            counter.update(tokenizer(string_.strip(), word))
    return Vocab(counter, min_freq=min_freq, specials=specials)


if __name__ == '__main__':
    filepath = 'data_01.txt'
    vocab = build_vocab(tokenizer, filepath, word=True, min_freq=1,
                        specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    print(vocab.freqs)  # 得到一个字典，返回语料中每个单词所出现的频率；
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['are'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))
