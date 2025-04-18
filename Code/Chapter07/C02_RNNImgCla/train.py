"""
文件名: Code/Chapter07/C02_RNNImgCla/train.py
创建时间: 2023/4/27 8:53 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from copy import deepcopy
import sys
import logging
import os

sys.path.append("../../")
from utils import logger_init
from FashionMNISTRNN import FashionMNISTRNN


class ModelConfig(object):
    def __init__(self):
        self.batch_size = 64
        self.epochs = 5
        self.learning_rate = 0.001
        self.num_classes = 10
        self.num_layers = 2
        self.input_size = 28
        self.hidden_size = 256
        self.model_save_path = 'model.pt'
        self.summary_writer_dir = "runs/model"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logger_init(log_file_name='log', log_level=logging.INFO, log_dir='log')
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def load_dataset(config, is_train=True):
    dataset = FashionMNIST(root='~/Datasets/FashionMNIST', train=is_train,
                           download=True, transform=transforms.ToTensor())
    iter = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      num_workers=1, pin_memory=False)  # 根据需要调整
    return iter


def train(config):
    train_iter = load_dataset(config, is_train=True)
    test_iter = load_dataset(config, is_train=False)
    model = FashionMNISTRNN(config.input_size, config.hidden_size,
                            config.num_layers, config.num_classes)
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型{config.model_save_path}进行追加训练...")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(config.summary_writer_dir)
    model = model.to(config.device)
    max_test_acc = 0
    global_steps = 0
    for epoch in range(config.epochs):
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(config.device), y.to(config.device)
            loss, logits = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 执行梯度下降
            global_steps += 1
            if i % 50 == 0:
                acc = (logits.argmax(1) == y).float().mean()
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--Acc: {round(acc.item(), 4)}--loss: {round(loss.item(), 4)}")
                writer.add_scalar('Training/Accuracy', acc, global_steps)
            writer.add_scalar('Training/Loss', loss.item(), global_steps)
        test_acc = evaluate(test_iter, model, config.device)
        logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--Acc on test {test_acc}")
        writer.add_scalar('Testing/Accuracy', test_acc, global_steps)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, config.model_save_path)


def evaluate(data_iter, model, device):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n


def inference(config, test_iter):
    model = FashionMNISTRNN(config.input_size, config.hidden_size,
                            config.num_layers, config.num_classes)
    model.to(config.device)
    model.eval()
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型进行推理……")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f" # 模型{config.model_save_path}不存在！")
    first_batch = next(iter(test_iter))
    with torch.no_grad():
        logits = model(first_batch[0].to(config.device))
    y_pred = logits.argmax(1)
    print(f"真实标签为：{first_batch[1]}")
    print(f"预测标签为：{y_pred}")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    test_iter = load_dataset(config, is_train=False)
    inference(config, test_iter)
    #  - INFO: [train.py][75] Epochs[1/5]--batch[0/938]--Acc: 0.0156--loss: 2.3238
    #  - INFO: [train.py][75] Epochs[1/5]--batch[50/938]--Acc: 0.4688--loss: 1.1348
    #  - INFO: [train.py][75] Epochs[1/5]--batch[100/938]--Acc: 0.5625--loss: 1.0453
    #  - INFO: [train.py][75] Epochs[1/5]--batch[150/938]--Acc: 0.7188--loss: 0.8871
    #  - INFO: [train.py][75] Epochs[1/5]--batch[200/938]--Acc: 0.6406--loss: 1.061
    #  - INFO: [train.py][75] Epochs[1/5]--batch[250/938]--Acc: 0.7344--loss: 0.7379
    # ......
    #  - INFO: [train.py][75] Epochs[5/5]--batch[600/938]--Acc: 0.8906--loss: 0.278
    #  - INFO: [train.py][75] Epochs[5/5]--batch[650/938]--Acc: 0.8906--loss: 0.2916
    #  - INFO: [train.py][75] Epochs[5/5]--batch[700/938]--Acc: 0.8438--loss: 0.4513
    #  - INFO: [train.py][75] Epochs[5/5]--batch[750/938]--Acc: 0.8438--loss: 0.4665
    #  - INFO: [train.py][75] Epochs[5/5]--batch[800/938]--Acc: 0.8438--loss: 0.4105
    #  - INFO: [train.py][75] Epochs[5/5]--batch[850/938]--Acc: 0.7969--loss: 0.5822
    #  - INFO: [train.py][75] Epochs[5/5]--batch[900/938]--Acc: 0.875--loss: 0.3218
    #  - INFO: [train.py][80] Epochs[5/5]--Acc on test 0.8622
