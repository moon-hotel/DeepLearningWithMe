"""
文件名: Code/Chapter08/C07_STResNet/STResNet.py
创建时间: 2023/6/19 19:18 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn


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
    def __init__(self, conv1_in_chs=8, conv1_out_chs=16, num_res_unit=3, res_out_chs=32, nb_flow=2):
        super().__init__()

        self.conv1 = nn.Conv2d(conv1_in_chs, conv1_out_chs, 3, stride=1, padding=1)
        res_units = []
        for i in range(num_res_unit):
            res_units.append(ResUnit(conv1_out_chs, res_out_chs))
        self.res_units = nn.ModuleList(res_units)
        self.conv2 = nn.Conv2d(conv1_out_chs, nb_flow, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for res_unit in self.res_units:
            x = res_unit(x)
        x = self.conv2(x)
        return x  #


class FeatureExt(nn.Module):
    def __init__(self, ext_dim=20, nb_flow=2, map_height=32, map_width=32):
        super().__init__()
        self.nb_flow = nb_flow
        self.map_height = map_height
        self.map_width = map_width

        self.feature = nn.Sequential(
            nn.Linear(ext_dim, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, nb_flow * map_height * map_width),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.feature(x.to(torch.float32))
        x = torch.reshape(x, [-1, self.nb_flow, self.map_height, self.map_width])
        return x


class STResNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.close = ResComponent(config.nb_flow * config.len_closeness,
                                  config.conv1_out_chs, config.num_res_unit, config.res_out_chs)
        self.period = ResComponent(config.nb_flow * config.len_period,
                                   config.conv1_out_chs, config.num_res_unit, config.res_out_chs)
        self.trend = ResComponent(config.nb_flow * config.len_trend,
                                  config.conv1_out_chs, config.num_res_unit, config.res_out_chs)
        self.ext_feature = FeatureExt(config.ext_dim, config.nb_flow, config.map_height, config.map_width)
        self.w_c = nn.Parameter(torch.randn([1, config.nb_flow, config.map_height, config.map_width]))
        self.w_p = nn.Parameter(torch.randn([1, config.nb_flow, config.map_height, config.map_width]))
        self.w_t = nn.Parameter(torch.randn([1, config.nb_flow, config.map_height, config.map_width]))

    def forward(self, x, y=None):
        x0 = self.close(x[0])
        x1 = self.period(x[1])
        x2 = self.trend(x[2])
        x3 = self.ext_feature(x[3])
        y1 = x0 * self.w_c + x1 * self.w_p + x2 * self.w_t
        logits = torch.tanh(y1 + x3)
        if y is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, y)
            return loss, logits
        else:
            return logits


class ModelConfig(object):
    def __init__(self):
        self.nb_flow = 2  #
        self.len_closeness = 3
        self.len_period = 1
        self.len_trend = 1
        self.conv1_out_chs = 64
        self.res_out_chs = 64
        self.num_res_unit = 12
        self.map_height = 32
        self.map_width = 32
        self.ext_dim = 28


if __name__ == '__main__':
    config = ModelConfig()
    x0 = torch.randn([16, config.nb_flow * config.len_closeness, config.map_height, config.map_width])
    x1 = torch.randn([16, config.nb_flow * config.len_period, config.map_height, config.map_width])
    x2 = torch.randn([16, config.nb_flow * config.len_trend, config.map_height, config.map_width])
    x3 = torch.randint(0, 2, size=[16, config.ext_dim])
    y = torch.randn([16, config.nb_flow, config.map_height, config.map_width])
    x = [x0, x1, x2, x3]
    st = STResNet(config)
    print(st(x).shape)
    loss, logits = st(x, y)
    print(logits.shape)
    print(loss)
    print(st)

    # torch.Size([16, 2, 32, 32])
    # torch.Size([16, 2, 32, 32])
    # tensor(1.4513, grad_fn=<MseLossBackward0>)
    # STResNet(
    #   (close): ResComponent(
    #     (conv1): Conv2d(6, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (res_units): ModuleList(
    #       (0-11): 12 x ResUnit(
    #         (block): Sequential(
    #           (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #           (1): ReLU(inplace=True)
    #           (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #           (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #           (4): ReLU(inplace=True)
    #           (5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         )
    #       )
    #     )
    #     (conv2): Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   )
    #   (period): ResComponent(
    #     (conv1): Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (res_units): ModuleList(
    #       (0-11): 12 x ResUnit(
    #         (block): Sequential(
    #           (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #           (1): ReLU(inplace=True)
    #           (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #           (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #           (4): ReLU(inplace=True)
    #           (5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         )
    #       )
    #     )
    #     (conv2): Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   )
    #   (trend): ResComponent(
    #     (conv1): Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (res_units): ModuleList(
    #       (0-11): 12 x ResUnit(
    #         (block): Sequential(
    #           (0): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #           (1): ReLU(inplace=True)
    #           (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #           (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    #           (4): ReLU(inplace=True)
    #           (5): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         )
    #       )
    #     )
    #     (conv2): Conv2d(64, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   )
    #   (ext_feature): FeatureExt(
    #     (feature): Sequential(
    #       (0): Linear(in_features=28, out_features=10, bias=True)
    #       (1): ReLU(inplace=True)
    #       (2): Linear(in_features=10, out_features=2048, bias=True)
    #       (3): ReLU(inplace=True)
    #     )
    #   )
    # )
