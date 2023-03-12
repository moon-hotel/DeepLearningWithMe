"""
文件名: Code/Chapter05/C05_ModelTrans/LeNet6.py
创建时间: 2023/3/7 8:05 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch.nn as nn
import torch
import os
from copy import deepcopy
import logging


class LeNet6(nn.Module):
    def __init__(self, ):
        super(LeNet6, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),  # [n,6,24,24]
            nn.MaxPool2d(2, 2),  # kernel_size, stride  [n,6,14,14]
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # [n,16,10,10]
            # PrintLayer(),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))  # [n,16,5,5]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16 * 5 * 5, out_features=120),
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=10))

    @classmethod
    def from_pretrained(cls, pretrained_model_dir=None, freeze=False):
        model = cls()
        frozen_list = []
        pretrained_model_path = os.path.join(pretrained_model_dir, "lenet5.pt")
        if not os.path.exists(pretrained_model_path):
            raise ValueError(f"<路径：{pretrained_model_path} 中的模型不存在，请仔细检查！>")
        loaded_paras = torch.load(pretrained_model_path)
        state_dict = deepcopy(model.state_dict())
        for key in state_dict:  # 在新的网络模型中遍历对应参数
            if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
                logging.info(f"成功初始化参数: {key}")
                state_dict[key] = loaded_paras[key]
                if freeze:
                    frozen_list.append(key)
        if len(frozen_list) > 0:
            for (name, param) in model.named_parameters():
                if name in frozen_list:
                    logging.info(f"冻结参数{name}")
                    param.requires_grad = False
        model.load_state_dict(state_dict)
        return model

    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    print("\n=====Model paras in LeNet6:")
    model = LeNet6()
    for (name, param) in model.state_dict().items():
        print(name, param.size())

    model_save_path = os.path.join('../C04_ModelSaving', 'lenet5.pt')
    print("\n=====Model paras in LeNet5:")
    loaded_paras = torch.load(model_save_path)
    for (name, param) in loaded_paras.items():
        print(name, param.size())
    print(f"LeNet5模型中第一层权重参数（部分）为：{loaded_paras['conv.0.weight'][0, 0]}")

    print("\n=====Load model from pretrained ")
    model = LeNet6.from_pretrained('../C04_ModelSaving', freeze=True)
    print(f"LeNet6模型中第一层权重参数（部分）为：{model.state_dict()['conv.0.weight'][0, 0]}")



    for (name, param) in model.named_parameters():
        print(name, param.size(), " --- is_trainable:", param.requires_grad)
