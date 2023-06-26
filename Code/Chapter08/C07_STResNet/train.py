"""
文件名: Code/Chapter08/C07_STResNet/train.py
创建时间: 2023/6/26 21:03 下午
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
from STResNet import STResNet
import torchvision.transforms as transforms

sys.path.append("../../")
from utils import TaxiBJ


class ModelConfig(object):
    def __init__(self):
        self.num_flow = 2  #
        self.len_closeness = 3
        self.len_period = 1
        self.len_trend = 1
        self.conv1_out_chs = 64
        self.res_out_chs = 128
        self.num_unit = 4
        self.map_height = 32
        self.map_width = 32
        self.ext_dim = 18
        self.batch_size = 64
        self.epochs = 50
        self.learning_rate = 4e-3
        self.num_warmup_steps = 300
        self.model_save_path = 'model.pt'
        self.summary_writer_dir = "runs/model"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    data_loader = TaxiBJ(len_test=config.len_test, len_closeness=config.len_closeness,
                         len_period=config.len_period, len_trend=config.len_trend)
    train_iter = data_loader.load_train_test_data(is_train=True)
    model = STResNet(config)
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
        for i, (XC_train, XP_train, XT_train, Y_train, _, _) in enumerate(train_iter):
            XC_train = XC_train.to(config.device)
            XP_train = XP_train.to(config.device)
            XT_train = XT_train.to(config.device)
            Y_train = Y_train.to(config.device)
            loss, logits = model([XC_train, XP_train, XT_train], Y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 执行梯度下降
            scheduler.step()
            if i % 50 == 0:
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--loss: {round(loss.item(), 4)}")
                writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
        # test_acc = evaluate(train_iter, model, config.device)
        # logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--Acc on val {test_acc}")
        # writer.add_scalar('Testing/Accuracy', test_acc, scheduler.last_epoch)
        # if test_acc > max_test_acc:
        #     max_test_acc = test_acc
        #     state_dict = deepcopy(model.state_dict())
        #     torch.save(state_dict, config.model_save_path)


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
    data_loader = TaxiBJ(len_test=config.len_test, len_closeness=config.len_closeness,
                         len_period=config.len_period, len_trend=config.len_trend)
    test_iter = data_loader.load_train_test_data(is_train=False)
    model = STResNet(config)
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
        logits = model(first_batch[0].permute(0, 2, 1, 3, 4).to(config.device))
    y_pred = logits.argmax(1)
    logging.info(f"真实标签为：{first_batch[1]}")
    logging.info(f"预测标签为：{y_pred}")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)

    #   推理
    # inference(config)
