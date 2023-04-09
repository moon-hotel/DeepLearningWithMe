"""
文件名: Code/Chapter06/C02_BN/train.py
创建时间: 2023/4/7 9:55 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from LeNet5 import LeNet5


def load_dataset():
    mnist_test = MNIST(root='~/Datasets/MNIST',
                       train=False, download=True,
                       transform=transforms.ToTensor())
    mnist_train = MNIST(root='~/Datasets/MNIST',
                        train=True, download=True,
                        transform=transforms.ToTensor())
    return mnist_train, mnist_test


def train(mnist_train, mnist_test):
    batch_size = 64
    learning_rate = 0.001
    epochs = 3
    model = LeNet5()
    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 定义优化器
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_iter):
            loss, logits = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # 执行梯度下降
            if i % 50 == 0:
                acc = (logits.argmax(1) == y).float().mean()
                print(f"Epochs[{epoch + 1}/{epochs}]--batch[{i}/{len(train_iter)}]"
                      f"--Acc: {round(acc.item(), 4)}--loss: {round(loss.item(), 4)}")
        print(f"Epochs[{epoch + 1}/{epochs}]--Acc on test {evaluate(test_iter, model)}")
        model.train()
    return model


def evaluate(data_iter, model):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            logits = model(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n


def inference(model, mnist_test):
    model.eval()
    y_true = mnist_test.targets[:5]
    imgs = mnist_test.data[:5].unsqueeze(1).to(torch.float32)
    with torch.no_grad():
        logits = model(imgs)
    y_pred = logits.argmax(1)
    print(f"真实标签为：{y_true}")
    print(f"预测标签为：{y_pred}")


if __name__ == '__main__':
    mnist_train, mnist_test = load_dataset()
    model = train(mnist_train, mnist_test)
    # inference(model, mnist_test)
# Epochs[1/3]--batch[0/938]--Acc: 0.0625--loss: 2.3926
# Epochs[1/3]--batch[50/938]--Acc: 0.9844--loss: 0.4578
# Epochs[1/3]--batch[100/938]--Acc: 0.9531--loss: 0.2169
# Epochs[1/3]--batch[150/938]--Acc: 0.9219--loss: 0.3134
# Epochs[1/3]--batch[200/938]--Acc: 0.9688--loss: 0.1778
# Epochs[1/3]--batch[250/938]--Acc: 0.9062--loss: 0.2766
# Epochs[1/3]--batch[300/938]--Acc: 0.9844--loss: 0.1543
# Epochs[1/3]--batch[350/938]--Acc: 0.9844--loss: 0.0652
# Epochs[1/3]--batch[400/938]--Acc: 0.9844--loss: 0.0431
# Epochs[1/3]--batch[450/938]--Acc: 0.9531--loss: 0.0962
# Epochs[1/3]--batch[500/938]--Acc: 1.0--loss: 0.031
# Epochs[1/3]--batch[550/938]--Acc: 0.9531--loss: 0.1236
# Epochs[1/3]--batch[600/938]--Acc: 0.9688--loss: 0.0838
# Epochs[1/3]--batch[650/938]--Acc: 0.9844--loss: 0.0682
# Epochs[1/3]--batch[700/938]--Acc: 0.9844--loss: 0.0692
# Epochs[1/3]--batch[750/938]--Acc: 0.9844--loss: 0.0669
# Epochs[1/3]--batch[800/938]--Acc: 1.0--loss: 0.0235
# Epochs[1/3]--batch[850/938]--Acc: 0.9844--loss: 0.0594
# Epochs[1/3]--batch[900/938]--Acc: 1.0--loss: 0.0322
# Epochs[1/3]--Acc on test 0.9808
