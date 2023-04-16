"""
文件名: Code/Chapter04/C09_DenseNet/DenseNet.py
创建时间: 2023/4/15 3:51 下午
"""

from torchvision.models.densenet import densenet121
import torch.nn as nn
import torch


class DenseBlock(nn.Module):
    """
    basic_block_coef: 论文中提及的1x1卷积到3x3卷积的输出通道数，论文里都是4*k即这里的4*growth_rate
    """

    def __init__(self, in_channels, growth_rate, basic_block_coef, drop_rate=0.5):
        super().__init__()
        self.block = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(in_channels, basic_block_coef * growth_rate,
                                             kernel_size=1, stride=1, bias=False),

                                   nn.BatchNorm2d(basic_block_coef * growth_rate),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(basic_block_coef * growth_rate, growth_rate,
                                             kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.Dropout(drop_rate))

    def forward(self, input):  # noqa: F811
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input
        concated_features = torch.cat(prev_features, 1)
        new_features = self.block(concated_features)
        return new_features


class DenseLayer(nn.Module):
    def __init__(self, num_dense_blocks, in_channels,
                 basic_block_coef, growth_rate, drop_rate=0.5):
        super().__init__()
        basic_blocks = []
        for i in range(num_dense_blocks):
            basic_blocks.append(DenseBlock(in_channels + i * growth_rate,
                                           growth_rate=growth_rate,
                                           basic_block_coef=basic_block_coef, drop_rate=drop_rate))
        self.basic_blocks = nn.Sequential(*basic_blocks)

    def forward(self, init_features):
        for basic_block in self.basic_blocks:
            out = basic_block(init_features)
            init_features = torch.cat((init_features, out), dim=1)
        return init_features


class Transition(nn.Module):
    """
    高宽通道均减半
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.transition = nn.Sequential(nn.BatchNorm2d(in_channels),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(in_channels, out_channels,
                                                  kernel_size=1, stride=1, bias=False),
                                        nn.AvgPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        return self.transition(x)


class DenseNet(nn.Module):
    def __init__(self, growth_rate: int = 32,
                 block_config=None,
                 num_init_features: int = 64, basic_block_coef=4,
                 drop_rate=0.5, num_classes=1000):
        super().__init__()

        # First convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        # Each DenseLayer
        num_features = num_init_features
        dense_layers = []
        for i, num_dense_blocks in enumerate(block_config):
            dense_layers.append(DenseLayer(num_dense_blocks=num_dense_blocks, in_channels=num_features,
                                           basic_block_coef=basic_block_coef, growth_rate=growth_rate,
                                           drop_rate=drop_rate))
            num_features = num_features + num_dense_blocks * growth_rate
            if i != len(block_config) - 1:
                dense_layers.append(Transition(in_channels=num_features, out_channels=int(num_features * 0.5)))
                num_features = int(num_features * 0.5)  # 0.5是论文中的Compression $\theta$
        self.dense_net = nn.Sequential(*dense_layers,
                                       nn.BatchNorm2d(num_features),  # Final batch norm
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool2d((1, 1)),
                                       nn.Flatten(),
                                       nn.Linear(num_features, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        out = self.dense_net(x)
        return out


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    # dense_layer = DenseLayer(num_dense_blocks=2, in_channels=3, basic_block_coef=4, growth_rate=16)
    # y = dense_layer(x)
    # print(dense_layer)
    # print(y.shape)
    #
    # transition_layer = Transition(in_channels=35, out_channels=8)
    # out_trans = transition_layer(y)
    # print(out_trans.shape)

    model = DenseNet(growth_rate=32, block_config=[2, 2, 2, 2],
                     num_init_features=64, basic_block_coef=4,
                     drop_rate=0.5, num_classes=1000)
    logits = model(x)
    print(model)
    print(logits.shape)

    for seq in model.children():
        for layer in seq:
            x = layer(x)
            print(f"网络层: {layer.__class__.__name__}, 输出形状: {x.shape}")
