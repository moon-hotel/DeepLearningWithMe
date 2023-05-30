"""
文件名: Code/Chapter07/C07_CharRNNPoetry/doPoetry.py
创建时间: 2023/5/16 8:44 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from CharRNN import CharRNN
from train import ModelConfig
import logging
import os
import torch
import sys

sys.path.append("../../")
from utils import TangShi


def greedy_decode(model, src, config, ends, UNK_IDX):
    """
    :param model:
    :param src: [1, src_len], 已经转换为token_ids
    :param num_sens: 句子数量
    :param config:
    :param ends: 结束符
    :return: [1,n]
    """
    max_len = [10 * config.num_sens, 12 * config.num_sens, 16 * config.num_sens]  # 四言、五言 或 七言 5*2+2， 7*2+2
    src = src.to(config.device)  # 初始时刻的输入形状通常应该是[1,1]，即只有一个字，但是也可以输入一句诗[1,n]
    for i in range(max(max_len) * 2):
        out = model(src)  # [1, src_len, vocab_size]
        if config.with_max_prob:
            _, next_word = torch.max(out[:, -1], dim=1)  # 每次在最后一个时刻的输出结果 中选择概率最大者
        else:
            prob = torch.softmax(out[:, -1], dim=-1)  # 计算得到最后一个时刻输出结果的概率分布
            while True:
                next_word = torch.distributions.Categorical(prob).sample()  # 根据概率分布采样得到下一个词
                if next_word.item() != UNK_IDX:
                    break
            # TODO: 这里还可以添加一些优化处理，例如预测结果如果是[UNK]或者[PAD]、或者是连续的句号逗号等等则取其它候选值等
        next_word = next_word.item()
        src = torch.cat([src, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        # 将当前时刻解码的预测输出结果，同之前所有的结果堆叠作为输入再去预测下一个词。
        if next_word in ends and (src.shape[1] in max_len or src.shape[1] > max(max_len)):
            break
    return src


def inference(config, srcs=None):
    model = CharRNN(config.top_k, config.embedding_size,
                    config.hidden_size, config.num_layers, config.cell_type,
                    config.bidirectional)
    model.to(config.device)
    model.eval()
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型进行推理……")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f" # 模型{config.model_save_path}不存在！")

    tang_shi = TangShi(top_k=config.top_k)
    srcs = tang_shi.make_infer_sample(srcs)
    unk_idx = tang_shi.vocab.stoi[tang_shi.vocab.UNK]
    with torch.no_grad():
        for src in srcs:
            result = greedy_decode(model, src, config, ends=tang_shi.ends, UNK_IDX=unk_idx)
            result = tang_shi.pretty_print(result)
            logging.info(f"\n{result}")


if __name__ == '__main__':
    config = ModelConfig()
    config.__dict__['num_sens'] = 4
    config.__dict__['with_max_prob'] = False
    srcs = ["李白乘舟将欲行", "朝辞"]
    inference(config, srcs)
