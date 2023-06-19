"""
文件名: Code/Chapter08/C07_STResNet/STResNet.py
创建时间: 2023/6/19 19:18 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUnit(nn.Module):
    def __init__(self, res_in_chs=16, res_out_chs=32):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(res_in_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_in_chs, res_out_chs, 3, stride=1, padding=1),
            nn.BatchNorm2d(res_out_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_out_chs, res_in_chs, 3, stride=1, padding=1))

    def forward(self, x):
        return x + self.block(x)


class ResComponent(nn.Module):
    def __init__(self, conv1_in_chs=8, conv1_out_chs=16, num_unit=3, res_out_chs=32, num_flow=2):
        super().__init__()

        self.conv1 = nn.Conv2d(conv1_in_chs, conv1_out_chs, 3, stride=1, padding=1)
        res_units = []
        for i in range(num_unit):
            res_units.append(ResUnit(conv1_out_chs, res_out_chs))
        self.res_units = nn.ModuleList(res_units)
        self.conv2 = nn.Conv2d(conv1_out_chs, num_flow, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.conv2(x)
        return x  #


class FeatureExt(nn.Module):
    def __init__(self, ext_dim=20, num_flow=2, map_height=32, map_width=32):
        super().__init__()
        self.num_flow = num_flow
        self.map_height = map_height
        self.map_width = map_width

        self.feature = nn.Sequential(
            nn.Linear(ext_dim, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, num_flow * map_height * map_width),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.feature(x.to(torch.float32))
        x = torch.reshape(x, [-1, self.num_flow, self.map_height, self.map_width])
        return x


class STResNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.trend = ResComponent(config.num_flow * config.len_closeness,
                                  config.conv1_out_chs, config.num_unit, config.res_out_chs)
        self.period = ResComponent(config.num_flow * config.len_period,
                                   config.conv1_out_chs, config.num_unit, config.res_out_chs)
        self.close = ResComponent(config.num_flow * config.len_trend,
                                  config.conv1_out_chs, config.num_unit, config.res_out_chs)
        self.w_t = nn.Parameter(torch.randn([1, config.num_flow, config.map_height, config.map_width]))
        self.w_p = nn.Parameter(torch.randn([1, config.num_flow, config.map_height, config.map_width]))
        self.w_c = nn.Parameter(torch.randn([1, config.num_flow, config.map_height, config.map_width]))
        self.ext_feature = FeatureExt(config.ext_dim, config.num_flow, config.map_height, config.map_width)

    def forward(self, x):
        x1 = self.trend(x[0])
        x2 = self.period(x[1])
        x3 = self.close(x[2])
        y1 = x1 * self.w_t + x2 * self.w_p + x3 * self.w_c
        y2 = self.ext_feature(x[3])
        y = F.tanh(y1 + y2)
        return y


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


if __name__ == '__main__':
    config = ModelConfig()
    x0 = torch.randn([16, config.num_flow * config.len_closeness, config.map_height, config.map_width])
    x1 = torch.randn([16, config.num_flow * config.len_period, config.map_height, config.map_width])
    x2 = torch.randn([16, config.num_flow * config.len_trend, config.map_height, config.map_width])
    x3 = torch.randint(0, 2, size=[16, config.ext_dim])
    x = [x0, x1, x2, x3]
    st = STResNet(config)
    print(st(x).shape)
