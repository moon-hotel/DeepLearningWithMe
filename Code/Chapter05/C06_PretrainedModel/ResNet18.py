"""
文件名: Code/Chapter05/C06_PretrainedModel/ResNet18.py
创建时间: 2023/4/15 5:23 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import logging

import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
import torch


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, frozen=False):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if frozen:
            for (name, param) in self.resnet18.named_parameters():
                param.requires_grad = False
                logging.info(f"冻结参数: {name}, {param.shape}")
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x, labels=None):
        logits = self.resnet18(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    model = ResNet18(frozen=True)
    x = torch.rand(1, 3, 96, 96)
    out = model(x)
    print(out)
    for (name, param) in model.named_parameters():
        print(f"name = {name,param.shape} requires_grad = {param.requires_grad}")
