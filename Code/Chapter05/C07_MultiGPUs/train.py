"""
文件名: Code/Chapter05/C07_MultiGPUs/train.py
创建时间: 2023/4/12 8:10 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import logging
import sys
import os
from copy import deepcopy

sys.path.append("../../")
from utils import logger_init
from utils import get_gpus
from Chapter04.C08_ResNet.ResNet import resnet18


class ModelConfig(object):
    def __init__(self, ):
        self.batch_size = 128
        self.epochs = 50
        self.learning_rate = 0.001
        self.in_channels = 3
        self.resize = 96
        self.num_classes = 10
        self.model_save_path = 'model.pt'
        self.summary_writer_dir = "runs/result"
        self.augment = True
        self.device = get_gpus(num=2)
        self.master_gpu_id = 0  # 主GPU
        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logger_init(log_file_name='train_log', log_level=logging.INFO, log_dir='log')
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def load_dataset(config, is_train=True):
    trans = [transforms.ToTensor()]
    if config.resize:  # 将输入的32*32的图片，resize成224*224的形状
        trans.append(transforms.Resize(size=config.resize,
                                       interpolation=InterpolationMode.BILINEAR))
    if config.augment and is_train:
        trans += [transforms.RandomHorizontalFlip(0.5),
                  transforms.CenterCrop(config.resize), ]
    trans = transforms.Compose(trans)
    dataset = CIFAR10(root='~/Datasets/CIFAR10', train=is_train,
                      download=True, transform=trans)
    iter = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      num_workers=1, pin_memory=False)  # 根据需要调整
    return iter


def train(config):
    train_iter = load_dataset(config, is_train=True)
    test_iter = load_dataset(config, is_train=False)
    model = resnet18(config.num_classes, config.in_channels)
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型{config.model_save_path}进行追加训练...")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    model = model.to(config.device[config.master_gpu_id])  # 指定主GPU
    model = nn.DataParallel(model, device_ids=config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(config.summary_writer_dir)
    max_test_acc = 0
    global_steps = 0
    for epoch in range(config.epochs):
        for i, (x, y) in enumerate(train_iter):
            x, y = x.to(config.device[config.master_gpu_id]), y.to(config.device[config.master_gpu_id])
            loss, logits = model(x, y)
            optimizer.zero_grad()
            loss.mean().backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()  # 执行梯度下降
            global_steps += 1
            if i % 50 == 0:
                acc = (logits.argmax(1) == y).float().mean()
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--Acc: {round(acc.item(), 4)}--loss: {round(loss.sum().item(), 4)}")
                writer.add_scalar('Training/Accuracy', acc, global_steps)
            writer.add_scalar('Training/Loss', loss.sum().item(), global_steps)
        test_acc = evaluate(test_iter, model, config.device[config.master_gpu_id])
        logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--Acc on test {test_acc}")
        writer.add_scalar('Testing/Accuracy', test_acc, global_steps)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            state_dict = deepcopy(model.module.state_dict())
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
    model = resnet18(config.num_classes, config.in_channels)
    model.to(config.device[config.master_gpu_id])
    model.eval()
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型进行推理……")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f" # 模型{config.model_save_path}不存在！")
    first_batch = next(iter(test_iter))
    with torch.no_grad():
        logits = model(first_batch[0].to(config.device[config.master_gpu_id]))
    y_pred = logits.argmax(1)
    print(f"真实标签为：{first_batch[1]}")
    print(f"预测标签为：{y_pred}")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    # test_iter = load_dataset(config, is_train=False)
    # inference(config, test_iter)
