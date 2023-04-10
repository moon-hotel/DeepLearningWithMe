"""
文件名: Code/Chapter04/C03_LeNet5/LeNet5.py
创建时间: 2023/3/22 8:49 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
from AlexNet import AlexNet
import logging
import sys
import os
from copy import deepcopy

sys.path.append("../../")
from utils import logger_init


class ModelConfig(object):
    def __init__(self):
        self.batch_size = 64
        self.epochs = 5
        self.learning_rate = 0.001
        self.in_channels = 1
        self.num_classes = 10
        self.resize = 224  # 将输入的28*28的图片，resize成224*224的形状
        self.augment = True
        self.model_save_path = 'alexnet.pt'
        self.summary_writer_dir = "runs/alexnet"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logger_init(log_file_name='alexnet', log_level=logging.INFO, log_dir='log')
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def load_dataset(config, is_train=True):
    trans = [transforms.ToTensor()]
    if config.resize is not None:  # 将输入的28*28的图片，resize成224*224的形状
        trans.append(transforms.Resize(size=config.resize,
                                       interpolation=InterpolationMode.BILINEAR))
    if config.augment and is_train:
        trans += [transforms.RandomHorizontalFlip(p=0.3),
                  transforms.ColorJitter(0.2, 0.3, 0.5, 0.5)]  # 以给定的概率随机水平翻转给定的图像
    trans = transforms.Compose(trans)
    dataset = FashionMNIST(root='~/Datasets/FashionMNIST', train=is_train,
                           download=True, transform=trans, )
    iter = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                      num_workers=1, pin_memory=False)
    # Tips: 提升GPU的利用率（通过nvidia-smi命令查看的Volatile GPU-Util字段）：
    # ①num_workers，意思是用多少个子进程加载数据，可以调成2，4，8，16等，但并不是越大越快，可自行调整；
    # ②pin_memory=True，锁页内存，内存够用一定要加上。
    # 主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存进行交换
    # （注：虚拟内存就是硬盘），而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。而显卡中的显存全部是锁页内存，
    # 设置为True时可以加快内存和显卡之前的数据交换；
    # https://zhuanlan.zhihu.com/p/477457147
    return iter


def train(config):
    train_iter = load_dataset(config, is_train=True)
    test_iter = load_dataset(config, is_train=False)
    model = AlexNet(config.in_channels, config.num_classes)
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
    model = AlexNet(config.in_channels, config.num_classes)
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
    # test_iter = load_dataset(config, is_train=False)
    # inference(config, test_iter)

    # 不进行图像增强
    # - INFO: [train.py][89] Epochs[1/5]--batch[0/938]--Acc: 0.1094--loss: 2.3021
    # - INFO: [train.py][89] Epochs[1/5]--batch[50/938]--Acc: 0.4531--loss: 1.4323
    # - INFO: [train.py][89] Epochs[1/5]--batch[100/938]--Acc: 0.6875--loss: 0.7821
    # ...
    # - INFO: [train.py][94] Epochs[1/5]--Acc on test 0.8503
    # - INFO: [train.py][94] Epochs[2/5]--Acc on test 0.8722
    # - INFO: [train.py][94] Epochs[3/5]--Acc on test 0.8921
    # - INFO: [train.py][94] Epochs[4/5]--Acc on test 0.9093
    # - INFO: [train.py][94] Epochs[5/5]--Acc on test 0.9064

    # 真实标签为: tensor([5, 1, 7, 0, 5, 8, 4, 1, 9, 5, 8, 8,...])
    # 预测标签为: tensor([5, 1, 7, 0, 5, 8, 4, 1, 9, 5, 8, 8,...])

    # 进行图像增强
    # - INFO: [train.py][89] Epochs[1/5]--batch[0/938]--Acc: 0.125--loss: 2.302
    # - INFO: [train.py][89] Epochs[1/5]--batch[50/938]--Acc: 0.4688--loss: 1.257
    # - INFO: [train.py][89] Epochs[1/5]--batch[100/938]--Acc: 0.7344--loss: 0.7736
    # ...
    # - INFO: [train.py][94] Epochs[1/5]--Acc on test 0.8375
    # - INFO: [train.py][94] Epochs[2/5]--Acc on test 0.8842
    # - INFO: [train.py][94] Epochs[3/5]--Acc on test 0.8911
    # - INFO: [train.py][94] Epochs[4/5]--Acc on test 0.9039
    # - INFO: [train.py][94] Epochs[5/5]--Acc on test 0.9024
