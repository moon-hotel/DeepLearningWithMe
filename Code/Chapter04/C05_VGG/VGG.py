"""
文件名: Code/Chapter04/C05_VGG/VGG.py
创建时间: 2023/3/26 9:48 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch.nn as nn
import torch

vgg_config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # 11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # 13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # 16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # 19
}  # M 表示maxpool


def make_layers(config):
    layers = []
    in_channels = config.in_channels
    cfg = vgg_config[config.vgg_type]
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # *号的作用解包这个list


class VGGNet(nn.Module):
    def __init__(self, features, config):
        super(VGGNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, config.num_classes))
        if config.init_weights:
            self._initialize_weights()

    def forward(self, x, labels=None):
        x = self.features(x)
        logits = self.classifier(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def vgg(config=None):
    cnn_features = make_layers(config)
    model = VGGNet(cnn_features, config)
    return model


class Config(object):
    def __init__(self):
        self.vgg_type = 'B'
        self.num_classes = 10
        self.init_weights = True
        self.in_channels = 3


if __name__ == '__main__':
    config = Config()
    vgg13 = vgg(config)
    print(vgg13)
    x = torch.rand(1, 3, 224, 224)
    y = vgg13(x)
    print(y)
    print(y.shape)
    # tensor([[-0.0017, -0.0261, -0.0275, -0.0067, -0.0419, -0.0358, -0.0022,  0.0367,
    #          -0.0127, -0.0105]], grad_fn=<AddmmBackward0>)
    # torch.Size([1, 10])
    no_layer = 0
    for seq in vgg13.children():
        for layer in seq:
            x = layer(x)
            if hasattr(layer, 'weight'):
                no_layer += 1
                print(f"网络层{no_layer}: {layer.__class__.__name__}, 输出形状: {x.shape}")
            else:
                print(f"网络层: {layer.__class__.__name__}, 输出形状: {x.shape}")

    # 网络层1: Conv2d, 输出形状: torch.Size([1, 64, 224, 224])
    # 网络层: ReLU, 输出形状: torch.Size([1, 64, 224, 224])
    # 网络层2: Conv2d, 输出形状: torch.Size([1, 64, 224, 224])
    # 网络层: ReLU, 输出形状: torch.Size([1, 64, 224, 224])
    # 网络层: MaxPool2d, 输出形状: torch.Size([1, 64, 112, 112])
    # 网络层3: Conv2d, 输出形状: torch.Size([1, 128, 112, 112])
    # 网络层: ReLU, 输出形状: torch.Size([1, 128, 112, 112])
    # 网络层4: Conv2d, 输出形状: torch.Size([1, 128, 112, 112])
    # 网络层: ReLU, 输出形状: torch.Size([1, 128, 112, 112])
    # 网络层: MaxPool2d, 输出形状: torch.Size([1, 128, 56, 56])
    # 网络层5: Conv2d, 输出形状: torch.Size([1, 256, 56, 56])
    # 网络层: ReLU, 输出形状: torch.Size([1, 256, 56, 56])
    # 网络层6: Conv2d, 输出形状: torch.Size([1, 256, 56, 56])
    # 网络层: ReLU, 输出形状: torch.Size([1, 256, 56, 56])
    # 网络层: MaxPool2d, 输出形状: torch.Size([1, 256, 28, 28])
    # 网络层7: Conv2d, 输出形状: torch.Size([1, 512, 28, 28])
    # 网络层: ReLU, 输出形状: torch.Size([1, 512, 28, 28])
    # 网络层8: Conv2d, 输出形状: torch.Size([1, 512, 28, 28])
    # 网络层: ReLU, 输出形状: torch.Size([1, 512, 28, 28])
    # 网络层: MaxPool2d, 输出形状: torch.Size([1, 512, 14, 14])
    # 网络层9: Conv2d, 输出形状: torch.Size([1, 512, 14, 14])
    # 网络层: ReLU, 输出形状: torch.Size([1, 512, 14, 14])
    # 网络层10: Conv2d, 输出形状: torch.Size([1, 512, 14, 14])
    # 网络层: ReLU, 输出形状: torch.Size([1, 512, 14, 14])
    # 网络层: MaxPool2d, 输出形状: torch.Size([1, 512, 7, 7])
    # 网络层: Flatten, 输出形状: torch.Size([1, 25088])
    # 网络层11: Linear, 输出形状: torch.Size([1, 4096])
    # 网络层: ReLU, 输出形状: torch.Size([1, 4096])
    # 网络层: Dropout, 输出形状: torch.Size([1, 4096])
    # 网络层12: Linear, 输出形状: torch.Size([1, 4096])
    # 网络层: ReLU, 输出形状: torch.Size([1, 4096])
    # 网络层: Dropout, 输出形状: torch.Size([1, 4096])
    # 网络层13: Linear, 输出形状: torch.Size([1, 10])