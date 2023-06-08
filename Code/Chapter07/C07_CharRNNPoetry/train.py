"""
文件名: Code/Chapter07/C07_CharRNNPoetry/train.py
创建时间: 2023/5/16 10:43 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import sys
import torch
import logging
from CharRNN import CharRNN
import os
from torch.utils.tensorboard import SummaryWriter
from transformers import optimization
from copy import deepcopy

sys.path.append("../../")
from utils import TangShi


class ModelConfig(object):
    def __init__(self):
        self.batch_size = 128
        self.epochs = 50
        self.learning_rate = 3e-3
        self.top_k = 2500  # vocab_size
        self.embedding_size = 256
        self.hidden_size = 512
        self.num_layers = 3
        self.cell_type = 'LSTM'
        self.max_len = None
        self.clip_max_norm = 0.8
        self.num_warmup_steps = 200
        self.model_save_path = 'model.pt'
        self.summary_writer_dir = "runs/model"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    tang_shi = TangShi(top_k=config.top_k, max_sen_len=config.max_len,
                       batch_size=config.batch_size)
    train_iter, val_iter = tang_shi.load_train_val_test_data(is_train=True)
    model = CharRNN(config.top_k, config.embedding_size, config.hidden_size,
                    config.num_layers, config.cell_type)
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型{config.model_save_path}进行追加训练...")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(config.summary_writer_dir)
    model = model.to(config.device)
    max_test_acc = 0
    steps = len(train_iter) * config.epochs
    scheduler = optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.num_warmup_steps,
                                                             num_training_steps=steps, num_cycles=2)
    for epoch in range(config.epochs):
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(config.device), y.to(config.device)
            loss, logits = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_max_norm)
            optimizer.step()  # 执行梯度下降
            scheduler.step()
            if i % 50 == 0:
                acc, _, _ = accuracy(logits, y, 1)
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--Acc: {round(acc, 4)}--loss: {round(loss.item(), 4)}")
                writer.add_scalar('Training/Accuracy', acc, scheduler.last_epoch)
            writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
        # test_acc = evaluate(val_iter, model, config.device)
        test_acc = evaluate(train_iter, model, config.device)
        logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--Acc on val {test_acc}")
        writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
        if test_acc > max_test_acc: # 因为
            max_test_acc = test_acc
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, config.model_save_path)


def evaluate(data_iter, model, device):
    model.eval()
    with torch.no_grad():
        corrects, totals = 0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, c, t = accuracy(logits, y)
            corrects += c
            totals += t
        model.train()
        return float(corrects) / totals


def accuracy(logits, y_true, PAD_IDX=1):
    """
    :param logits:  [batch_size,src_len,vocab_size]
    :param y_true:  [batch_size,tgt_len]
    :param PAD_IDX:
    :return:
    """
    y_pred = logits.argmax(axis=2).reshape(-1)
    y_true = y_true.reshape(-1)
    acc = y_pred.eq(y_true)  # 计算预测值与正确值比较的情况
    mask = torch.logical_not(y_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    acc = acc.logical_and(mask)  # 去掉acc中mask的部分
    correct = acc.sum().item()
    total = mask.sum().item()
    return float(correct) / total, correct, total


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
