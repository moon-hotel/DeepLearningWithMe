"""
文件名: Code/Chapter07/C07_CharRNNPoetry/doPoetry.py
创建时间: 2023/5/16 8:44 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from CharRNN import CharRNN
import logging
import os
import torch
import sys

sys.path.append("../../")
from utils import TangShi


def greedy_decode(model, src, config):
    """

    :param model:
    :param src: [batch_size, src_len], 已经转换为token_ids
    :param num_sens: 句子数量
    :param config:
    :return:
    """
    max_len = [5 * config.num_sens, 7 * config.num_sens]  # 五言 或 七言
    src = src.to(config.device)  # 初始时刻的输入形状通常应该是[1,1]，即只有一个字，但是也可以输入一句诗
    for i in range(max(max_len) - 1):
        out = model(src)  # [1,src_len,vocab_size]
        _, next_word = torch.max(out[:, -1], dim=1)  # 每次在最后一个时刻的输出结果 中选择概率最大者
        next_word = next_word.item()
        src = torch.cat([src, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        # 将当前时刻解码的预测输出结果，同之前所有的结果堆叠作为输入再去预测下一个词。
        if len(src.shape[1]) in max_len:
            break
    return src


def inference(config, srcs=None):
    model = CharRNN(config.vocab_size, config.embedding_size,
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
    with torch.no_grad():
        for src in srcs:
            result = greedy_decode(model, src, config)
            print(result)

    # y_pred = logits.argmax(1)
# print(f"真实标签为：{first_batch[1]}")
# print(f"预测标签为：{y_pred}")
