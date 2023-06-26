"""
文件名: Code/Chapter08/C05_ConvLSTM/train.py
创建时间: 2023/6/10 7:18 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from transformers import optimization
from torch.utils.tensorboard import SummaryWriter
import torch
from copy import deepcopy
import sys
import logging
import os
from ConvLSTM import ConvLSTMKTH
import torchvision.transforms as transforms

sys.path.append("../../")
from utils import KTHData


class ModelConfig(object):
    def __init__(self):
        self.batch_size = 32
        self.epochs = 30
        self.learning_rate = 3e-3
        self.num_classes = 6
        self.in_channels = 1
        self.out_channels = [32,32]
        self.kernel_size = [(3, 3), (3, 3)]
        self.num_layers = len(self.out_channels)
        self.height = 60  # 原始大小为120
        self.width = 80  # 原始大小为160
        self.time_step = 15
        self.num_warmup_steps = 200
        self.model_save_path = 'model.pt'
        self.summary_writer_dir = "runs/model"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.height, config.width)),
        transforms.RandomHorizontalFlip(0.5)])
    data_load = KTHData(frame_len=config.time_step,
                        batch_size=config.batch_size,
                        transforms=trans)
    train_iter, val_iter = data_load.load_train_val_test_data(is_train=True)
    model = ConvLSTMKTH(config)
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
            optimizer.step()  # 执行梯度下降
            scheduler.step()
            if i % 50 == 0:
                acc = (logits.argmax(1) == y).float().mean()
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--Acc: {round(acc.item(), 4)}--loss: {round(loss.item(), 4)}")
                writer.add_scalar('Training/Accuracy', acc, scheduler.last_epoch)
            writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
        test_acc = evaluate(val_iter, model, config.device)
        logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--Acc on val {test_acc}")
        writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
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


def inference(config, ):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((config.height, config.width)),
        transforms.RandomHorizontalFlip(0.5)])
    data_load = KTHData(frame_len=config.time_step,
                        batch_size=config.batch_size,
                        transforms=trans)
    test_iter = data_load.load_train_val_test_data(is_train=False)
    model = ConvLSTMKTH(config)
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
    logging.info(f"真实标签为：{first_batch[1]}")
    logging.info(f"预测标签为：{y_pred}")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
    #   推理
    # inference(config)