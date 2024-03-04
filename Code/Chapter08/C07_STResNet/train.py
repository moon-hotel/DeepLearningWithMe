"""
文件名: Code/Chapter08/C07_STResNet/train.py
创建时间: 2023/6/26 21:03 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import numpy as np
from transformers import optimization
from torch.utils.tensorboard import SummaryWriter
import torch
from copy import deepcopy
import sys
import logging
import os
from STResNet import STResNet

sys.path.append("../../")
from utils import TaxiBJ
from utils import logger_init

class ModelConfig(object):
    def __init__(self):
        self.nb_flow = 2  #
        self.len_closeness = 3
        self.len_period = 1
        self.len_trend = 1
        self.conv1_out_chs = 64
        self.res_out_chs = 64
        self.num_res_unit = 4
        self.map_height = 32
        self.map_width = 32
        self.ext_dim = 28
        self.T = 48
        self.len_test = self.T * 4 * 7  # 4周一共28天作为测试集
        self.batch_size = 64
        self.epochs = 50
        self.learning_rate = 4e-3
        self.num_warmup_steps = 300
        self.model_save_path = 'model.pt'
        self.summary_writer_dir = "runs/model"
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 判断是否存在GPU设备，其中0表示指定第0块设备
        logger_init(log_file_name='log', log_level=logging.INFO, log_dir='log')
        logging.info("### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    data_loader = TaxiBJ(config.T, config.nb_flow, config.len_test, config.len_closeness,
                         config.len_period, config.len_trend, batch_size=config.batch_size)
    train_iter, mmn = data_loader.load_train_test_data(is_train=True)
    model = STResNet(config)
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型{config.model_save_path}进行追加训练...")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    writer = SummaryWriter(config.summary_writer_dir)
    model = model.to(config.device)
    steps = len(train_iter) * config.epochs
    scheduler = optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.num_warmup_steps,
                                                             num_training_steps=steps, num_cycles=2)
    max_train_loss = 10000.
    for epoch in range(config.epochs):
        total_loss = 0
        for i, (XC, XP, XT, Y, meta_feature, _) in enumerate(train_iter):
            XC = XC.to(config.device)  # [batch_size, 2*len_closeness, 32,32]
            XP = XP.to(config.device)  # [batch_size, 2*len_period, 32,32]
            XT = XT.to(config.device)  # [batch_size, 2*len_trend, 32,32]
            Y = Y.to(config.device)  # [batch_size, 2, 32,32]
            meta_feature = meta_feature.to(config.device)
            loss, logits = model([XC, XP, XT, meta_feature], Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 执行梯度下降
            scheduler.step()
            total_loss += loss.item()
            if i % 50 == 0:
                logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--batch[{i}/{len(train_iter)}]"
                             f"--loss: {round(loss.item(), 4)}")
                writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
        logging.info(f"Epochs[{epoch + 1}/{config.epochs}] --Total loss: {total_loss}")
        if total_loss < max_train_loss:
            max_train_loss = total_loss
            state_dict = deepcopy(model.state_dict())
            torch.save(state_dict, config.model_save_path)
            rmse = evaluate(train_iter, model, config.device, mmn)
            logging.info(f"Epochs[{epoch + 1}/{config.epochs}]--RMSE on train: {round(rmse, 4)}")
        # inference(config)


def compute_rmse(all_logits=None, all_labels=None, mmn=None):
    all_logits = torch.cat(all_logits, dim=0)  # [n,2,32,32]
    all_labels = torch.cat(all_labels, dim=0)  # [n,2,32,32]
    y_pred = all_logits.detach().cpu().numpy()  # [1344, 2,32,32]
    y_true = all_labels.detach().cpu().numpy()  # [1344, 2,32,32]
    y_pred = mmn.inverse_transform(y_pred)
    y_true = mmn.inverse_transform(y_true)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return rmse


def evaluate(data_iter, model, device, mmn):
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for i, (XC, XP, XT, Y, meta_feature_test, _) in enumerate(data_iter):
            XC = XC.to(device)
            XP = XP.to(device)
            XT = XT.to(device)
            Y = Y.to(device)
            meta_feature_test = meta_feature_test.to(device)
            loss, logits = model([XC, XP, XT, meta_feature_test], Y)
            all_logits.append(logits)  # [[batch_size,2,32,32], [batch_size,2,32,32],...]
            all_labels.append(Y)  # [[batch_size,2,32,32], [batch_size,2,32,32],...]
        model.train()
        rmse = compute_rmse(all_logits, all_labels, mmn)
    return rmse


def inference(config):
    data_loader = TaxiBJ(config.T, config.nb_flow, config.len_test, config.len_closeness,
                         config.len_period, config.len_trend, batch_size=config.batch_size)
    test_iter, mmn = data_loader.load_train_test_data(is_train=False)
    model = STResNet(config)
    model.to(config.device)
    model.eval()
    if os.path.exists(config.model_save_path):
        logging.info(f" # 载入模型进行推理……")
        checkpoint = torch.load(config.model_save_path)
        model.load_state_dict(checkpoint)
    else:
        raise ValueError(f" # 模型{config.model_save_path}不存在！")
    rmse = evaluate(test_iter, model, config.device, mmn)
    logging.info(f" # RMSE on test: {rmse}")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)

    #   推理
    # inference(config)
