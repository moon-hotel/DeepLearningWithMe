"""
文件名: Code/Chapter04/C06_NIN/NIN.py
创建时间: 2023/3/28 8:59 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import torch.nn as nn
import torch


def nin_block(in_chs, out_chs=None, k_size=5, s=1, p=2):
    nin_seq = nn.Sequential(
        nn.Conv2d(in_chs, out_chs[0], k_size, s, p),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chs[0], out_chs[1], 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chs[1], out_chs[2], 1, 1),
        nn.ReLU(inplace=True))
    return nin_seq


class NIN(nn.Module):
    def __init__(self, init_weights=True):
        super(NIN, self).__init__()

        self.nin = nn.Sequential(
            nin_block(in_chs=3, out_chs=[192, 160, 96], k_size=5, s=1, p=2),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(0.5),
            nin_block(in_chs=96, out_chs=[192, 192, 192], k_size=5, s=1, p=2),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Dropout(0.5),
            nin_block(in_chs=192, out_chs=[192, 192, 10], k_size=3, s=1, p=1),
            nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        if init_weights:
            self._initialize_weights()

    def forward(self, x, labels=None):
        logits = self.nin(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    x = torch.rand(1, 3, 224, 224)
    model = NIN()
    # print(model)
    y = model(x)

    no_layer = 0
    for seq in model.children():
        for layer in seq:
            x = layer(x)
            if hasattr(layer, 'weight'):
                no_layer += 1
                print(f"网络层{no_layer}: {layer.__class__.__name__}, 输出形状: {x.shape}")
            else:
                print(f"网络层: {layer.__class__.__name__}, 输出形状: {x.shape}")
