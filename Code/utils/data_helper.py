"""
文件名: Code/utils/data_helper.py
创建时间: 2023/5/6 8:07 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import json
import os
import torch
import logging
import h5py
import time

from tqdm import tqdm
import numpy as np
from copy import copy
from PIL import Image
from torch.utils.data import DataLoader
from collections import Counter
import matplotlib.pyplot as plt
import unicodedata
import pandas as pd
from gensim import utils
from .tools import process_cache
from .tools import MinMaxNormalization
from .tools import timestamp2vec
from .tools import string2timestamp
from .tools import contains_chinese

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
    :param data: 为一个列表，每个元素为一句文本
    :return:
    """
    UNK = '[UNK]'  # 0
    PAD = '[PAD]'  # 1

    def __init__(self, top_k=2000, data=None, show_distribution=False, cut_words=False):
        logging.info(f" ## 正在根据训练集构建词表……")
        counter = Counter()
        self.stoi = {Vocab.UNK: 0, Vocab.PAD: 1}
        self.itos = [Vocab.UNK, Vocab.PAD]
        for text in data:
            token = tokenize(text, cut_words)
            # ['上', '联', '：', '一', '夜', '春', '风', '去', '，', '怎', '么', '对', '下', '联', '？']
            counter.update(token)  # 统计每个token出现的频率
        if show_distribution:
            count_w = sorted(list(counter.values()))
            plt.scatter(range(len(count_w)), count_w[::-1], s=5)
            plt.ylim(-20, 2500)
            plt.show()
        top_k_words = counter.most_common(top_k - 2)  # 取前top_k - 2 个，加上UNK和PAD，一共top_k个
        for i, word in enumerate(top_k_words):
            self.stoi[word[0]] = i + 2  # 2表示已有了UNK和PAD
            self.itos.append(word[0])
        logging.info(f" ## 词表构建完毕，前100个词为: {list(self.stoi.items())[:100]}")

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def tokenize(text, cut_words=False):
    """
    tokenize方法
    :param text: 上联：一夜春风去，怎么对下联？
    :return:
    words: 字粒度： ['上', '联', '：', '一', '夜', '春', '风', '去', '，', '怎', '么', '对', '下', '联', '？']
    """
    import jieba
    if contains_chinese(text):  # 中文
        if cut_words:  # 分词
            text = jieba.cut(text)  # 词粒度
        text = " ".join(text)  # 不分词则是字粒度
    words = text.split()
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
    训练集:验证集:测试集 267881:76537:8270
    """
    DATA_DIR = os.path.join(DATA_HOME, 'toutiao')
    FILE_PATH = [os.path.join(DATA_DIR, 'toutiao_train.txt'),
                 os.path.join(DATA_DIR, 'toutiao_val.txt'),
                 os.path.join(DATA_DIR, 'toutiao_test.txt')]

    def __init__(self, top_k=2000,
                 max_sen_len=None,
                 batch_size=4,
                 is_sample_shuffle=True,
                 cut_words=False):
        self.top_k = top_k
        self.cut_words = cut_words
        raw_data_train, _ = self.load_raw_data(self.FILE_PATH[0])
        self.vocab = Vocab(top_k=self.top_k, data=raw_data_train, cut_words=self.cut_words)
        self.max_sen_len = max_sen_len
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def get_vocab(self):
        return self.vocab.stoi

    @staticmethod
    def load_raw_data(file_path=None):
        """
        载入原始的文本
        :param file_path:
        :return:
        samples: ['上联：一夜春风去，怎么对下联？', '隋唐五代｜欧阳询《温彦博碑》，书于贞观十一年，是欧最晚的作品']
        labels: ['1','1']
        """
        logging.info(f" ## 载入原始文本 {file_path.split(os.path.sep)[-1]}")
        samples, labels = [], []
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n').split('_!_')
                samples.append(line[0])
                labels.append(line[1])
        return samples, labels

    @process_cache(unique_key=["top_k", "cut_words", "max_sen_len", "is_sample_shuffle"])
    def data_process(self, file_path=None):
        samples, labels = self.load_raw_data(file_path)
        data = []
        logging.info(f" ## 处理原始文本 {file_path.split(os.path.sep)[-1]}")
        for i in tqdm(range(len(samples)), ncols=80):
            logging.debug(f" ## 原始输入样本为: {samples[i]}")
            tokens = tokenize(samples[i], self.cut_words)
            logging.debug(f" ## 分割后的样本为: {tokens}")
            token_ids = [self.vocab[token] for token in tokens]
            logging.debug(f" ## 向量化后样本为: {token_ids}\n")
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
            l = torch.tensor(int(labels[i]), dtype=torch.long)
            data.append((token_ids_tensor, l))
        return data

    def generate_batch(self, data_batch):
        """
        以小批量为单位对序列进行padding
        :param data_batch:
        :return:
        """
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
        if not is_train:
            test_data = self.data_process(file_path=self.FILE_PATH[2])
            test_iter = DataLoader(test_data, batch_size=self.batch_size,
                                   shuffle=False, collate_fn=self.generate_batch)
            logging.info(f" ## 测试集构建完毕，一共{len(test_data)}个样本")
            return test_iter
        train_data = self.data_process(file_path=self.FILE_PATH[0])  # 得到处理好的所有样本
        val_data = self.data_process(file_path=self.FILE_PATH[1])
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        logging.info(f" ## 训练集和验证集构建完毕，样本数量为{len(train_data)}:{len(val_data)}")
        return train_iter, val_iter


class TangShi(TouTiaoNews):
    """
    全唐诗数据集：
    感谢此仓库收集与整理 https://github.com/chinese-poetry/chinese-poetry
    此处使用的是全唐诗数据，即poet.tang.0.json~poet.tang.57000.json，58个json文件，一共约5.7万首

    """
    DATA_DIR = os.path.join(DATA_HOME, 'peotry_tang')
    FILE_PATH = [os.path.join(DATA_DIR, 'poet.tang.0-55.json'),  # 0~55000 for train
                 os.path.join(DATA_DIR, 'poet.tang.56-56.json'),  # 56000~56000 for val
                 os.path.join(DATA_DIR, 'poet.tang.57-57.json')]  # 57000~57000 for test

    def __init__(self, *args, **kwargs):
        super(TangShi, self).__init__(*args, **kwargs)
        self.ends = [self.vocab.stoi["。"], self.vocab.stoi["？"]]
        self.cut_words = False

    def load_raw_data(self, file_path=None):
        """
        :param file_path:
        :return:
        """

        def read_json_data(path):
            logging.info(f" ## 载入原始文本 {path.split(os.path.sep)[-1]}")
            samples, labels = [], []
            with open(path, encoding='utf-8') as f:
                data = json.loads(f.read())
                # [{'author': '范成大', 'paragraphs': ['日滿東窗照被堆，宿湯猶自暖如煨。', '尺三汗脚君休笑，曾踏鞾霜待漏來。'], 'title': '戲贈脚婆',...},
                # {'author': '范成大', 'paragraphs': ['雪不成花夜雨來，壠頭一洗定無埃。', '小童却怕溪橋滑，明日先生合探梅。'], 'title': '除夜前二日夜雨', ...},
                for item in data:
                    content = item['paragraphs']  # ['日滿東窗照被堆，宿湯猶自暖如煨。', '尺三汗脚君休笑，曾踏鞾霜待漏來。']
                    content = "".join(content)  # '日滿東窗照被堆，宿湯猶自暖如煨。尺三汗脚君休笑，曾踏鞾霜待漏來。'
                    if not skip(content):
                        samples.append(content[:-1])  # ['','日滿東窗照被堆，宿湯猶自暖如煨', '尺三汗脚君休笑，曾踏鞾霜待漏來']
                        labels.append(content[1:])  # 向左平移 ['','滿東窗照被堆，宿湯猶自暖如煨。尺三汗脚君休笑，曾踏鞾霜待漏來。']
                    else:
                        logging.debug(f"过滤古诗：len = {len(content)}, {content}")
            return samples, labels

        def skip(content):
            """
            过滤掉不需要的诗，里面可能含有一些杂乱的信息
            :param content:
            :return:
            """
            # if len(content) % 10 != 0 and len(content) % 12 != 0\
            #         and len(content) % 16 != 0:  # 四言、五言 或 七言
            #     return True

            if len(content) < 12 or len(content) > 100:  # 太长太短的诗过滤
                return True
            if '《' in content or '（' in content or '□' in content or '[' in content:  #
                return True
            return False

        file_name = file_path.split(os.path.sep)[-1]
        start, end = file_name.split('.')[2].split('-')
        all_samples, all_labels = [], []
        for i in range(int(start), int(end) + 1):
            # 按文件名中的数字索引构建原始文件的路径
            file_path = os.path.join(self.DATA_DIR, f'poet.tang.{i * 1000}.json')
            samples, labels = read_json_data(file_path)  # 读取每一个原始json文件
            all_samples += samples  # 累计所有样本
            all_labels += labels
        logging.info(f" ## {file_name} 样本数量为: {len(all_samples)}")
        return all_samples, all_labels

    @process_cache(unique_key=["top_k", "cut_words", "max_sen_len", "is_sample_shuffle"])
    def data_process(self, file_path=None):
        samples, labels = self.load_raw_data(file_path)
        data = []
        logging.info(f" ## 处理原始文本 {file_path.split(os.path.sep)[-1]}")
        for i in tqdm(range(len(samples)), ncols=80):
            logging.debug(f" ## 原始样本为:\n ")
            logging.debug(f" ## 输入为: {samples[i]}")
            x_tokens = tokenize(samples[i])
            logging.debug(f" ## 分割后为: {x_tokens}")
            x_token_ids = [self.vocab[token] for token in x_tokens]
            logging.debug(f" ## 向量化后为: {x_token_ids}\n")

            logging.debug(f" ## 标签为: {labels[i]}")
            y_tokens = tokenize(labels[i])
            logging.debug(f" ## 分割后为: {y_tokens}")
            y_token_ids = [self.vocab[token] for token in y_tokens]
            logging.debug(f" ## 向量化后为: {y_token_ids}\n")

            x_token_ids_tensor = torch.tensor(x_token_ids, dtype=torch.long)
            y_token_ids_tensor = torch.tensor(y_token_ids, dtype=torch.long)
            data.append((x_token_ids_tensor, y_token_ids_tensor))
        return data

    def generate_batch(self, data_batch):
        """
        以小批量为单位对序列进行padding
        :param data_batch:
        :return:
        """
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            batch_label.append(label)
        x_batch_sentence = pad_sequence(batch_sentence,  # [batch_size,max_len]
                                        padding_value=self.vocab.stoi[self.vocab.PAD],
                                        batch_first=True,
                                        max_len=self.max_sen_len)
        y_batch_sentence = pad_sequence(batch_label,  # [batch_size,max_len]
                                        padding_value=self.vocab.stoi[self.vocab.PAD],
                                        batch_first=True,
                                        max_len=self.max_sen_len)
        return x_batch_sentence, y_batch_sentence

    @staticmethod
    def simplified_traditional_convert(text, type='s2t'):
        """
        由于原始文本为繁体，所以需要对繁简体相互转换
        安装命令： pip install opencc-python-reimplemented
        :param text:
        :param type: t2s 繁体转简体， s2t简体转繁体
        :return:
        """
        import opencc
        if type not in ['t2s', 's2t']:
            raise ValueError(" ## 转换类型必须为 't2s' or 's2t'")
        converter = opencc.OpenCC(type)  # 使用t2s.json配置文件进行繁->简转换
        converted_text = converter.convert(text)
        return converted_text

    def make_infer_sample(self, srcs):
        """
        :param srcs: ["李白乘舟将欲行","朝辞白帝彩"]
        :return: [tensor([[767,  32, 388, 214, 113, 108,  34]]), tensor([[ 69, 366,  32, 390, 720]])]
        """
        all_token_ids = []
        logging.info(f" ## 构造推理样本")
        for src in srcs:
            text = self.simplified_traditional_convert(src, 's2t')
            tokens = tokenize(text)
            logging.info(f" ## 分割后为: {tokens}")
            token_ids = [self.vocab[token] for token in tokens]
            logging.info(f" ## 向量化后为: {token_ids}")
            token_ids = torch.tensor(token_ids, dtype=torch.long)
            all_token_ids.append(torch.reshape(token_ids, [1, -1]))
        return all_token_ids

    def pretty_print(self, result):
        """
        格式化输出结果
        :param result:  token id , [1,n]
        result = torch.tensor([[773, 217, 898, 122, 17, 2, 215, 23, 286, 16, 63, 3, 74, 428, 1897, 1112, 58, 2, 21, 15, 493,
        5, 269, 3, 723, 10, 19, 6, 48, 2, 869, 863, 4, 153, 1605, 3, 16, 46, 556, 25, 219, 1034, 88, 89, 78, 45, 1188, 3]])
        :return:
        借问陇头水，终年恨何事。
        深疑呜咽声，中有征人泪。
        昨日上山下，达曙不能寐。
        何处接长波？东流入清渭。
        """
        result = [self.vocab.itos[item.item()] for item in result[0]]
        result = "".join(result)
        result = self.simplified_traditional_convert(result, 't2s')
        seps = [self.vocab.itos[idx] for idx in self.ends]
        for sep in seps:
            result = result.split(sep)
            result = f"{sep}\n".join(result)
        result = result.split('\n')
        true_result = [result[0]]
        i = 1
        while i < len(result) - 1:
            if len(result[i]) < len(result[i - 1]):
                true_result.append(result[i] + result[i + 1])
                i += 2
            else:
                true_result.append(result[i])
                i += 1
        true_result = "\n".join(true_result)
        return true_result


class MR(TouTiaoNews):
    DATA_DIR = os.path.join(DATA_HOME, 'MR')
    FILE_PATH = [os.path.join(DATA_DIR, 'rt_train.txt'),
                 os.path.join(DATA_DIR, 'rt_val.txt'),
                 os.path.join(DATA_DIR, 'rt_test.txt')]


class MR4ELMo(TouTiaoNews):
    DATA_DIR = os.path.join(DATA_HOME, 'MR')
    FILE_PATH = [os.path.join(DATA_DIR, 'rt_train.txt'),
                 os.path.join(DATA_DIR, 'rt_val.txt'),
                 os.path.join(DATA_DIR, 'rt_test.txt')]

    def __init__(self, batch_size=32, is_sample_shuffle=True):
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle

    def data_process(self, file_path=None):
        samples, labels = self.load_raw_data(file_path)
        data = []
        logging.info(f" ## 处理原始文本 {file_path.split(os.path.sep)[-1]}")
        for i in tqdm(range(len(samples)), ncols=80):
            logging.debug(f" ## 原始输入样本为: {samples[i]}")
            data.append((samples[i].split(), labels[i]))
        return data

    def generate_batch(self, data_batch):
        """
        :param data_batch:
        :return:
        """
        from allennlp.modules.elmo import batch_to_ids
        batch_sentence, batch_label = [], []
        for (sen, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            batch_sentence.append(sen)
            l = torch.tensor(int(label), dtype=torch.long)
            batch_label.append(l)
        batch_sentence = batch_to_ids(batch_sentence)  # [batch_size, seq_len, 50]
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_sentence, batch_label


class KTHData(object):
    """
    载入KTH数据集，下载地址：https://www.csc.kth.se/cvap/actions/ 一共包含6个zip压缩包
    """
    DATA_DIR = os.path.join(DATA_HOME, 'kth')
    CATEGORIES = ["boxing", "handclapping", "handwaving", "jogging", "running", "walking"]
    TRAIN_PEOPLE_ID = [1, 2, 4, 5, 6, 7, 9, 11, 12, 15, 17, 18, 20, 21, 22, 23, 24]  # 25*0.7 = 17
    VAL_PEOPLE_ID = [3, 8, 10, 19, 25]  # 25*0.2 = 5
    TEST_PEOPLE_ID = [13, 14, 16]  # 25*0.1 = 3
    FILE_PATH = os.path.join(DATA_DIR, 'kth.pt')

    def __init__(self, frame_len=15,
                 batch_size=4,
                 is_sample_shuffle=True,
                 is_gray=True,
                 transforms=None):
        self.frame_len = frame_len  # 即time_step， 以FRAME_LEN为长度进行分割
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle
        self.is_gray = is_gray
        self.transforms = transforms

    @staticmethod
    def load_avi_frames(path=None, is_gray=False):
        """
        用来读取每一个.avi格式的文件
        :param path:
        :return:
        """
        import cv2
        logging.info(f" ## 正在读取原始文件: {path}并划分数据")
        video = cv2.VideoCapture(path)
        frames = []
        while video.isOpened():
            ret, frame = video.read()  # frame: (120, 160, 3) <class 'numpy.ndarray'>
            if not ret:  # ret是一个布尔值，表示是否成功读取帧图像的数据，frame是读取到的帧图像数据。
                break
            if is_gray:
                frame = Image.fromarray(frame)
                frame = frame.convert("L")
                frame = np.array(frame.getdata()).reshape((120, 160, 1))
            frames.append(frame)
        logging.info(f" ## 该视频一共有{len(frames)}帧")
        return np.array(frames, dtype=np.uint8)  # [n, height, width, channels]
        # 必须要转换成np.uint8类型，否则transforms.ToTensor()中的标准化会无效

    @process_cache(unique_key=["frame_len", "is_gray"])
    def data_process(self, file_path=None):
        train_data, val_data, test_data = [], [], []
        for label, dir_name in enumerate(self.CATEGORIES):  # 遍历每个文件夹
            video_dir = os.path.join(self.DATA_DIR, dir_name)  # 构造每个文件夹的路径
            video_names = os.listdir(video_dir)  # 列出当前文件夹的所有文件
            for name in video_names:  # 遍历当前文件夹中的每个视频
                people_id = int(name[6:8])  # 取人员编号
                video_path = os.path.join(video_dir, name)  # 得到文件的绝对路径
                frames = self.load_avi_frames(video_path, self.is_gray)  # 读取该文件
                s_idx, e_idx = 0, self.frame_len
                while e_idx <= len(frames):  # 开始采样样本
                    logging.info(f" ## 截取帧子序列 [{s_idx}:{e_idx}]")
                    sub_frames = frames[s_idx:e_idx]  # [frame_len, 120, 160, channels]
                    if people_id in self.TRAIN_PEOPLE_ID:
                        train_data.append((sub_frames, label))
                    elif people_id in self.VAL_PEOPLE_ID:
                        val_data.append((sub_frames, label))
                    elif people_id in self.TEST_PEOPLE_ID:
                        test_data.append((sub_frames, label))
                    else:
                        raise ValueError(f"people id {people_id} 有误")
                    s_idx, e_idx = e_idx, e_idx + self.frame_len
        logging.info(f" ## 原始数据划分完毕，训练集、验证集和测试集的数量分别为: "
                     f"{len(train_data)}-{len(val_data)}-{len(test_data)}")
        data = {"train_data": train_data, "val_data": val_data, "test_data": test_data}
        return data

    def generate_batch(self, data_batch):
        """
        :param data_batch:
        :return: 每个batch的形状
                 [batch_size, frame_len, channels, height, width]
                 [batch_size, ]
        """
        batch_frames, batch_label = [], []
        for (frames, label) in data_batch:  # 开始对一个batch中的每一个样本进行处理。
            # frames的形状为 [frame_len, height, width,channels]
            if self.transforms is not None:
                # 遍历序列里的每一帧，frame的形状[height, width, channels]
                # 经过transforms.ToTensor()后的形状为[channels, height, width]
                frames = torch.stack([self.transforms(frame) for frame in frames],
                                     dim=0)  # [frame_len, channels, height, width]
            else:
                frames = torch.tensor(frames.transpose(0, 3, 1, 2))  # [frame_len, channels, height, width]
                logging.info(f"{frames.shape}")
            batch_frames.append(frames)  # [[frame_len, channels, height, width], [], []]
            batch_label.append(label)
        batch_frames = torch.stack(batch_frames, dim=0)  # [batch_size, frame_len, channels, height, width]
        batch_label = torch.tensor(batch_label, dtype=torch.long)
        return batch_frames, batch_label

    def load_train_val_test_data(self, is_train=False):
        data = self.data_process(file_path=self.FILE_PATH)
        if not is_train:
            test_data = data['test_data']
            test_iter = DataLoader(test_data, batch_size=self.batch_size,
                                   shuffle=True, collate_fn=self.generate_batch)
            logging.info(f" ## 测试集构建完毕，一共{len(test_data)}个样本")
            return test_iter
        train_data, val_data = data['train_data'], data['val_data']
        train_iter = DataLoader(train_data, batch_size=self.batch_size,  # 构造DataLoader
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=False, collate_fn=self.generate_batch)
        logging.info(f" ## 训练集和验证集构建完毕，样本数量为{len(train_data)}:{len(val_data)}")
        return train_iter, val_iter

    def show_example(self, file_path=None, row=3, col=5, begin_id=10):
        """
        可视化
        :param file_path:
        :param row:
        :param col:
        :param begin_id:
        :return:
        """
        import matplotlib.pyplot as plt
        if file_path is None:
            file_path = os.path.join(self.DATA_DIR, self.CATEGORIES[0])
            file_path = os.path.join(file_path, 'person01_boxing_d1_uncomp.avi')
        frames = self.load_avi_frames(file_path)
        fig, ax = plt.subplots(row, col)
        for i, axi in enumerate(ax.flat):  # , figsize=(18, 10)
            image = frames[i + begin_id]
            axi.set_xlabel(f'Frame{i + begin_id}')
            axi.imshow(image)
            axi.set(xticks=[], yticks=[])
        plt.tight_layout()
        plt.show()


class STMatrix(object):
    """docstring for STMatrix
    构造采样数据帧
    """

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps  # [b'2013070101', b'2013070102']
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()  # 将时间戳：做成一个字典，也就是给每个时间戳一个序号

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            logging.info(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):  # 给定时间戳返回对于的数据
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_trend=3, TrendInterval=7, len_period=3, PeriodInterval=1):
        """
        :param len_closeness:
        :param len_trend:
        :param TrendInterval: 趋势性的间隔天数，默认为1周，即7天
        :param len_period:
        :param PeriodInterval: 周期性的间隔天数，默认为1天
        :return:
        """

        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)  # 时间偏移 minutes = 30
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []
        depends = [range(1, len_closeness + 1),
                   [PeriodInterval * self.T * j for j in range(1, len_period + 1)],
                   [TrendInterval * self.T * j for j in range(1, len_trend + 1)]]
        # depends # [range(1, 4), [48, 96, 144], [336, 672, 1008]]
        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            # 取当前时刻的前3个时间片的数据数据构成“邻近性”模块中一个输入序列
            # 例如当前时刻为[Timestamp('2013-07-01 00:00:00')]
            # 则取：
            # [Timestamp('2013-06-30 23:30:00'), Timestamp('2013-06-30 23:00:00'), Timestamp('2013-06-30 22:30:00')]
            #  三个时刻所对应的in-out flow为一个序列
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            # 取当前时刻 前 1*PeriodInterval,2*PeriodInterval,...,len_period*PeriodInterval
            # 天对应时刻的in-out flow 作为一个序列，例如按默认值为 取前1、2、3天同一时刻的In-out flow
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            # 取当前时刻 前 1*TrendInterval,2*TrendInterval,...,len_trend*TrendInterval
            # 天对应时刻的in-out flow 作为一个序列,例如按默认值为 取 前7、14、21天同一时刻的In-out flow
            y = self.get_matrix(self.pd_timestamps[i])
            if len_closeness > 0:
                XC.append(np.vstack(x_c))
                # a.shape=[2,32,32] b.shape=[2,32,32] c=np.vstack((a,b)) -->c.shape = [4,32,32]
            if len_period > 0:
                XP.append(np.vstack(x_p))
            if len_trend > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])  # []
            i += 1
        XC = np.asarray(XC)  # 模拟 邻近性的 数据 [?,6,32,32]
        XP = np.asarray(XP)  # 模拟 周期性的 数据 隔天
        XT = np.asarray(XT)  # 模拟 趋势性的 数据 隔周
        Y = np.asarray(Y)  # [?,2,32,32]
        logging.info(f"XC shape: {XC.shape}, XP shape: {XP.shape}, XT shape: {XT.shape} , Y shape: {Y.shape}")
        return XC, XP, XT, Y, timestamps_Y


class TaxiBJ(object):
    """
    载入北京出租车数据集，数据集可关注微信公众号@月来客栈 获取
    此代码修改自郑宇老师团队开源的ST-ResNet模型
    """
    DATA_DIR = os.path.join(DATA_HOME, 'TaxiBJ')
    FILE_PATH_FLOW = [os.path.join(DATA_DIR, 'BJ13_M32x32_T30_InOut.h5'),
                      os.path.join(DATA_DIR, 'BJ14_M32x32_T30_InOut.h5'),
                      os.path.join(DATA_DIR, 'BJ15_M32x32_T30_InOut.h5'),
                      os.path.join(DATA_DIR, 'BJ16_M32x32_T30_InOut.h5')]
    FILE_PATH_HOLIDAY = os.path.join(DATA_DIR, 'BJ_Holiday.txt')
    FILE_PATH_METEORO = os.path.join(DATA_DIR, 'BJ_Meteorology.h5')
    CATH_FILE_PATH = os.path.join(DATA_DIR, 'TaxiBJ.pt')

    def __init__(self, T=48, nb_flow=2, len_test=None, len_closeness=None,
                 len_period=None, len_trend=None, meta_data=True,
                 meteorol_data=True, holiday_data=True, batch_size=4, is_sample_shuffle=True):
        self.T = T
        self.nb_flow = nb_flow
        self.len_test = len_test
        self.len_closeness = len_closeness
        self.len_period = len_period
        self.len_trend = len_trend
        self.meta_data = meta_data
        self.meteorology_data = meteorol_data
        self.holiday_data = holiday_data
        self.batch_size = batch_size
        self.is_sample_shuffle = is_sample_shuffle
        assert len_closeness > 0, "len_closeness 需要大于0"
        assert len_period > 0, "len_period 需要大于0"
        assert len_trend > 0, "len_trend 需要大于0"

    def load_holiday(self, timeslots=None):
        """
        加载节假日列表，并返回给定时间戳中哪些日期是节假日，哪些不是
        :param timeslots: 字符串形式的时间戳，如: 2018120106
        :param filepath:
        :return: [[1],[1],[0],[0],[0]...] 当前时间片对应为假期则为1,否则为0
        e.g. load_holiday(timeslots=['2014120106','2014120206','2014010106','2014120706'])
            [[0.]
             [0.]
             [1.] 元旦为节假日
             [0.]] # shape:(4,1)
        """
        filepath = self.FILE_PATH_HOLIDAY
        with open(filepath, 'r') as f:
            holidays = f.readlines()
            holidays = set([h.strip() for h in holidays])
            # 得到一个假期列表，形如：['20130101', '20130102', '20130103', '20130209', ...]
            H = np.zeros(len(timeslots))
            for i, slot in enumerate(timeslots):
                if slot[:8] in holidays:  # 取前8位为日期，判断其是否为节假日
                    H[i] = 1
        return H[:, None]  # shape: [n,1]

    def load_meteorology(self, timeslots=None):
        """
        In real-world, we don't have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots,
        i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
        载入气象数据
        :param timeslots: 字符串形式的时间戳
        :return:
        """
        file_path = self.FILE_PATH_METEORO
        with h5py.File(file_path, 'r') as f:
            Timeslot = f['date'][:]  # 时间片
            WindSpeed = f['WindSpeed'][:]  # 风速
            Weather = f['Weather'][:]  # 天气
            Temperature = f['Temperature'][:]  # 温度
        M = dict()  # map timeslot to index
        for i, slot in enumerate(Timeslot):
            # 给每个时间戳赋一个索引
            M[slot] = i  # {...,b'2016061335': 59003, b'2016061336': 59004, b'2016061337': 59005}

        WS = []  # WindSpeed
        WR = []  # Weather
        TE = []  # Temperature
        for slot in timeslots:
            predicted_id = M[slot]  # 取索引
            cur_id = predicted_id - 1  # 取上一个索引，因为一般来说预测第t时刻时只能取其t-1时刻的天气信息
            WS.append(WindSpeed[cur_id])  #
            WR.append(Weather[cur_id])
            TE.append(Temperature[cur_id])
        WS = np.asarray(WS)  # shape: (n,)
        WR = np.asarray(WR)  # shape: (n,)
        TE = np.asarray(TE)  # shape: (n,)

        # 0-1 scale
        # 这里是一次对所有的温度和风速进行标准化，严格来说应该是需要划分乘训练集和测试集之后再标准化
        WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
        TE = 1. * (TE - TE.min()) / (TE.max() - TE.min())
        #
        logging.info(f"meteorol shape: {WS.shape, WR.shape, TE.shape}")
        # concatenate all these attributes
        merge_data = np.hstack([WR, WS[:, None], TE[:, None]])  # [n,19]
        # (n,17) (n,1) (n,1) ==> (n,19)
        logging.info(f'meger shape:{merge_data.shape}')
        return merge_data

    @staticmethod
    def load_stdata(fname):
        """
        载入原始数据
        split the data and date(timestamps)
        :param fname:
        :return:
        """
        f = h5py.File(fname, 'r')
        data = f['data'][:]
        timestamps = f['date'][:]
        f.close()
        return data, timestamps

    @staticmethod
    def stat(fname):
        """
        统计数据信息
        count the valid data
        :param fname:
        :return: like below

        ==========stat==========
        data shape: (7220, 2, 32, 32)
        # of days: 162, from 2015-11-01 to 2016-04-10
        # of timeslots: 7776
        # of timeslots (available): 7220
        missing ratio of timeslots: 7.2%
        max: 1250.000, min: 0.000
        ==========stat==========

        """

        def get_nb_timeslot(f):
            """
            count the number of timeslot of given data
            :param f:
            :return:
            """
            s = f['date'][0]
            e = f['date'][-1]
            year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
            ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
            year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
            te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
            nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
            time_s_str, time_e_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
            return nb_timeslot, time_s_str, time_e_str

        with h5py.File(fname) as f:
            nb_timeslot, time_s_str, time_e_str = get_nb_timeslot(f)
            nb_day = int(nb_timeslot / 48)
            mmax = f['data'][:].max()
            mmin = f['data'][:].min()
            stat = '=' * 10 + 'stat' + '=' * 10 + '\n' + \
                   '\tdata shape: %s\n' % str(f['data'].shape) + \
                   '\t# of days: %i, from %s to %s\n' % (nb_day, time_s_str, time_e_str) + \
                   '\t# of timeslots: %i\n' % int(nb_timeslot) + \
                   '\t# of timeslots (available): %i\n' % f['date'].shape[0] + \
                   '\tmissing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
                   '\tmax: %.3f, min: %.3f\n' % (mmax, mmin) + \
                   '\t' + '=' * 10 + 'stat' + '=' * 10
            logging.info(f"\n\t{stat}")

    @staticmethod
    def remove_incomplete_days(data, timestamps, T=48):
        """
        remove a certain day which has not 48 timestamps
        :param data:
        :param timestamps:
        :param T:
        :return:
        """
        days = []  # available days: some day only contain some seqs
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
                days.append(timestamps[i][:8])
                i += T
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        logging.info(f"Incomplete days: {days_incomplete}")
        days = set(days)
        idx = []
        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]
        return data, timestamps

    @process_cache(unique_key=["T", "nb_flow", "len_test", "len_closeness",
                               "len_period", "meta_data", "meteorology_data", "holiday_data"])
    def data_process(self, file_path=None):
        data_all = []
        timestamps_all = list()
        for fname in self.FILE_PATH_FLOW:
            logging.info(f" # 正在载入文件: {fname}")
            self.stat(fname)
            data, timestamps = self.load_stdata(fname)
            data, timestamps = self.remove_incomplete_days(data, timestamps, self.T)
            # data: ndarray  shape: [num, 2, 32, 32]
            # timestamps: [b'2013070101', b'2013070102', b'2013070103',...]
            data = data[:, :self.nb_flow]
            logging.info(data.shape)
            data[data < 0] = 0.  # 处理异常，把小于0的数据替换为0
            data_all.append(data)  # 保存每个文件读取处理完成后的结果
            timestamps_all.append(timestamps)
            logging.info("\n")
        # minmax_scale
        data_train = np.vstack(copy(data_all))[:-self.len_test]  # 划分出训练集部分
        logging.info(f'train data shape: {data_train.shape}')
        mmn = MinMaxNormalization()
        mmn.fit(data_train)  # 在训练集上计算相关参数
        data_all_mmn = [mmn.transform(d) for d in data_all]  # 依次对所有数据用训练集中计算得到的参数进行标准化
        logging.info(f"timestamps_all示例: {timestamps_all[0][:10]}")
        XC, XP, XT = [], [], []
        Y = []
        timestamps_Y = []
        for data, timestamps in zip(data_all_mmn, timestamps_all):  # 遍历4个文件中每个文件里的流量数据
            # instance-based dataset --> sequences with format as (X, Y) where X is
            # a sequence of images and Y is an image.
            st = STMatrix(data, timestamps, self.T, CheckComplete=False)  # 采样构造流量数据
            _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset(
                len_closeness=self.len_closeness, len_period=self.len_period, len_trend=self.len_trend)
            XC.append(_XC)
            XP.append(_XP)
            XT.append(_XT)
            Y.append(_Y)
            timestamps_Y += _timestamps_Y  # [ b'2013102232', b'2013102233', b'2013102234', b'2013102235',......]
        meta_feature = []
        if self.meta_data:
            # load time feature
            time_feature = timestamp2vec(timestamps_Y)  # array: [?,8] 将字符串类型的时间换为表示星期几和工作日的向量
            meta_feature.append(time_feature)
        if self.holiday_data:
            # load holiday
            holiday_feature = self.load_holiday(timestamps_Y)  # array: [?,1]加载节假日列表，并返回给定时间戳中那些日期是节假日，哪些不是
            meta_feature.append(holiday_feature)
        if self.meteorology_data:
            # load meteorol data
            meteorol_feature = self.load_meteorology(timestamps_Y)  # array: [?,19] 载入气象数据
            meta_feature.append(meteorol_feature)

        meta_feature = np.hstack(meta_feature) if len(
            meta_feature) > 0 else np.asarray(meta_feature)
        metadata_dim = meta_feature.shape[1] if len(
            meta_feature.shape) > 1 else None
        if metadata_dim < 1:
            metadata_dim = None
        if self.meta_data and self.holiday_data and self.meteorology_data:
            logging.info(f' ## time feature: {time_feature.shape}, holiday feature: {holiday_feature.shape},'
                         f'meteorol feature: {meteorol_feature.shape} mete feature: {meta_feature.shape}')
            ## time feature: (15072, 8), holiday feature: (15072, 1),meteorol feature: (15072, 19) mete feature: (15072, 28)
        XC = torch.tensor(np.vstack(XC), dtype=torch.float32)  # shape = [15072,6,32,32]
        XP = torch.tensor(np.vstack(XP), dtype=torch.float32)  # shape = [15072,2,32,32]
        XT = torch.tensor(np.vstack(XT), dtype=torch.float32)  # shape = [15072,2,32,32]
        Y = torch.tensor(np.vstack(Y), dtype=torch.float32)  # shape = [15072,2,32,32]
        meta_feature = torch.tensor(meta_feature, dtype=torch.float32)  # shape =[15072, 28]
        timestamps_Y = [str(item) for item in timestamps_Y]
        XC_train, XP_train, XT_train, Y_train = \
            XC[:-self.len_test], XP[:-self.len_test], XT[:-self.len_test], Y[:-self.len_test]
        XC_test, XP_test, XT_test, Y_test = \
            XC[-self.len_test:], XP[-self.len_test:], XT[-self.len_test:], Y[-self.len_test:]
        timestamp_train, timestamp_test = timestamps_Y[:-self.len_test], timestamps_Y[-self.len_test:]
        meta_feature_train, meta_feature_test = meta_feature[:-self.len_test], meta_feature[-self.len_test:]
        logging.info(f"数据集构建完毕")
        logging.info("训练集形状分别为:\n "
                     f"XC_train: {XC_train.shape}\n"
                     f"XP_train: {XP_train.shape}\n"
                     f"XT_train: {XT_train.shape}\n"
                     f"Y_train: {Y_train.shape}\n"
                     f"meta_feature_train: {meta_feature_train.shape}\n"
                     f"timestamp_train: {len(timestamp_train)}\n")
        logging.info("测试集形状分别为:\n "
                     f"XC_test: {XC_test.shape}\n"
                     f"XP_test: {XP_test.shape}\n"
                     f"XT_test: {XT_test.shape}\n"
                     f"Y_test: {Y_test.shape}\n"
                     f"meta_feature_test: {meta_feature_test.shape}\n"
                     f"timestamp_test: {len(timestamp_test)}\n")
        train_data = [item for item in zip(XC_train, XP_train, XT_train, Y_train, meta_feature_train, timestamp_train)]
        test_data = [item for item in zip(XC_test, XP_test, XT_test, Y_test, meta_feature_test, timestamp_test)]
        data = {"train_data": train_data, "test_data": test_data, "mmn": mmn}
        return data

    def load_train_test_data(self, is_train=False):
        data = self.data_process(file_path=self.CATH_FILE_PATH)
        mmn = data['mmn']
        if not is_train:
            test_data = data['test_data']
            test_iter = DataLoader(test_data, batch_size=self.batch_size, shuffle=True)
            logging.info(f" ## 测试集构建完毕，一共{len(test_data)}个样本")
            return test_iter, mmn
        train_data = data['train_data']
        train_iter = DataLoader(train_data, batch_size=self.batch_size, shuffle=self.is_sample_shuffle)
        logging.info(f" ## 训练集构建完毕，样本数量为{len(train_data)}")
        return train_iter, mmn

    def show_example(self, num_id=100):
        """
        可视化示例
        :param num_id:  样本编号
        :return:
        """
        data, _ = self.load_stdata(self.FILE_PATH_FLOW[0])
        plt.imshow(data[num_id][1], cmap='RdYlGn_r', interpolation='nearest')
        plt.colorbar()
        plt.show()


class SougoNews(object):
    DATA_DIR = os.path.join(DATA_HOME, 'SougoNews')
    PROCESSED_FILE_PATH = os.path.join(DATA_DIR, 'SougoNews.txt')

    def __init__(self, ):
        self.make_corpus()

    def make_corpus(self):
        self.corpus_path = self.PROCESSED_FILE_PATH
        if not os.path.exists(self.PROCESSED_FILE_PATH):
            logging.info(f" ## 预处理文件{self.PROCESSED_FILE_PATH}不存在，即将重新读取文件生成！")
            for i in range(5, 1, -1):
                logging.info(f"Countdown: {i}s")
                time.sleep(1)
            self.data_process()

    def data_process(self, ):
        dir_lists = os.listdir(self.DATA_DIR)  # 列出当前文件夹中的所有文件夹
        num_file = 0
        num_failed = 0
        new_file = open(self.PROCESSED_FILE_PATH, 'w', encoding='utf-8')
        for dir in dir_lists:
            dir_name = os.path.join(self.DATA_DIR, dir)  # 构造得到每个目录的路径
            file_lists = os.listdir(dir_name)
            logging.info(f" ## 正在读取文件夹【{dir_name}】中的文件")
            for file in tqdm(file_lists):
                file_path = os.path.join(dir_name, file)
                logging.debug(f" ## 正在读取第{num_file + 1}个文件：{file_path}")
                num_file += 1
                result = []
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        for line in f:
                            line = line.strip().replace('&nbsp', '')
                            if len(line) < 30:
                                continue
                            line = unicodedata.normalize('NFKC', line)  # 将全角字符转换为半角字符
                            seg = tokenize(line, cut_words=True)
                            result += seg
                except:
                    num_failed += 1
                new_file.write(" ".join(result) + '\n')
        new_file.close()
        logging.info(f"一共读取文件个数为: {num_file}, 读取失败个数为: {num_failed}")


class MyCorpus(SougoNews):
    """An iterator that yields sentences (lists of str)."""

    def __init__(self):
        super(MyCorpus, self).__init__()
        pass

    def __iter__(self):
        logging.info(f" ## 读取预处理文件进行训练: {self.PROCESSED_FILE_PATH}")
        for line in open(self.corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)
