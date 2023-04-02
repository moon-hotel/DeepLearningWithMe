import torch
import torch.nn as nn

config = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}  # M表示maxpool


def make_layers(cfg):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # *号的作用解包这个list


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes))
        if init_weights:
            self._initialize_weights()

    def forward(self, x, labels=None):
        x = self.features(x)
        x = self.avgpool(x)
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


def vgg_11(num_classes=1000, init_weights=True):
    cnn_features = make_layers(config['A'])
    model = VGG(cnn_features, num_classes=num_classes, init_weights=init_weights)
    return model


def vgg_13(num_classes=1000, init_weights=True):
    cnn_features = make_layers(config['B'])
    model = VGG(cnn_features, num_classes=num_classes, init_weights=init_weights)
    return model


def vgg_16(num_classes=1000, init_weights=True):
    cnn_features = make_layers(config['D'])
    model = VGG(cnn_features, num_classes=num_classes, init_weights=init_weights)
    return model


def vgg_19(num_classes=1000, init_weights=True):
    cnn_features = make_layers(config['E'])
    model = VGG(cnn_features, num_classes=num_classes, init_weights=init_weights)
    return model


if __name__ == '__main__':
    vgg11 = vgg_11()
    print(vgg11)
