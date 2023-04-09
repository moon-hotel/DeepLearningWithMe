"""
文件名: Code/Chapter06/C02_GradClip/train.py
创建时间: 2023/4/9 7:53 下午
"""

from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
import sys

sys.path.append('../../')
from Chapter04.C03_LeNet5.LeNet5 import LeNet5


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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
            optimizer.step()  # 执行梯度下降
            if i % 50 == 0:
                acc = (logits.argmax(1) == y).float().mean()
                print(f"Epochs[{epoch + 1}/{epochs}]--batch[{i}/{len(train_iter)}]"
                      f"--Acc: {round(acc.item(), 4)}--loss: {round(loss.item(), 4)}")
        print(f"Epochs[{epoch + 1}/{epochs}]--Acc on test {evaluate(test_iter, model)}")
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
    inference(model, mnist_test)
