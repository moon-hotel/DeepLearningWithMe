"""
文件名: Code/Chapter06/C04_ModelSaving/train.py
创建时间: 2023/3/5 7:52 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import torch
import sys
import logging

sys.path.append("../../")
from Chapter04.C03_LeNet5.LeNet5 import LeNet5
from Chapter05.C02_LogManage.log_manage import logger_init
from copy import deepcopy
import os
import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def load_dataset(batch_size=64, only_test=False):
    fashion_test = FashionMNIST(root='~/Datasets/FashionMNIST', train=False,
                                download=True, transform=transforms.ToTensor())
    test_iter = DataLoader(fashion_test, batch_size=batch_size, shuffle=True)
    if only_test:
        return test_iter
    fashion_train = FashionMNIST(root='~/Datasets/FashionMNIST', train=True,
                                 download=True, transform=transforms.ToTensor())
    train_iter = DataLoader(fashion_train, batch_size=batch_size, shuffle=True)
    return train_iter, test_iter


class ModelConfig(object):
    def __init__(self,
                 batch_size=64,
                 epochs=3,
                 learning_rate=0.01):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model_save_path = 'model.pt'
        self.summary_writer_dir = "runs/lenet5"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger_init(log_file_name='lenet5', log_level=logging.INFO, log_dir='log')
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    train_iter, test_iter = load_dataset(config.batch_size)
    model = LeNet5()

    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)

    num_training_steps = len(train_iter) * config.epochs
    optimizer = torch.optim.Adam([{"params": model.parameters(),
                                   "initial_lr": config.learning_rate}])
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=300,
                                                num_training_steps=num_training_steps,
                                                num_cycles=2)
    writer = SummaryWriter(config.summary_writer_dir)
    model = model.to(config.device)
    max_test_acc = 0
    model.train()
    for epoch in range(config.epochs):
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(config.device), y.to(config.device)
            loss, logits = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 执行梯度下降
            scheduler.step()
            if i % 50 == 0:
                acc = (logits.argmax(1) == y).float().mean()
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--Acc: {round(acc.item(), 4)}--loss: {round(loss.item(), 4)}")
                writer.add_scalar('Training/Accuracy', acc, scheduler.last_epoch)
            writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
            writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)

        test_acc, all_logits, y_labels, label_img = evaluate(test_iter, model, config.device)
        logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--Acc on test {test_acc}")
        writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
        writer.add_embedding(mat=all_logits,  # 所有点
                             metadata=y_labels,  # 标签名称
                             label_img=label_img,  # 标签图片
                             global_step=scheduler.last_epoch)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, config.model_save_path)
    return model


def evaluate(data_iter, model, device):
    model.eval()
    all_logits = []
    y_labels = []
    images = []
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)

            all_logits.append(logits)
            y_pred = logits.argmax(1).view(-1)
            y_labels += (text_labels[i] for i in y_pred)
            images.append(x)
        return acc_sum / n, torch.cat(all_logits, dim=0), y_labels, torch.cat(images, dim=0)


def inference(config, test_iter):
    test_data = test_iter.dataset
    model = LeNet5()
    if os.path.exists(config.model_save_path):
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"模型{config.model_save_path}不存在！")
    y_true = test_data.targets[:5]
    imgs = test_data.data[:5].unsqueeze(1).to(torch.float32)
    with torch.no_grad():
        logits = model(imgs)
    y_pred = logits.argmax(1)
    print(f"真实标签为：{y_true}")
    print(f"预测标签为：{y_pred}")


if __name__ == '__main__':
    config = ModelConfig()
    # model = train(config)
    test_iter = load_dataset(only_test=True)
    inference(config, test_iter)
