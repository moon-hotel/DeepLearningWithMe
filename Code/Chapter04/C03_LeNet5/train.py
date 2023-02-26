"""
文件名: Code/Chapter04/C03_LeNet5/train.py
创建时间: 2023/2/26 9:55 上午
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
    model.train()
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
    return model


def evaluate(data_iter, model):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            logits = model(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        return acc_sum / n


def inference(model, mnist_test):
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

# Epochs[1/3]--batch[0/938]--Acc: 0.0938--loss: 2.2998
# Epochs[1/3]--batch[50/938]--Acc: 0.7344--loss: 0.9657
# Epochs[1/3]--batch[100/938]--Acc: 0.8594--loss: 0.506
# Epochs[1/3]--batch[150/938]--Acc: 0.9219--loss: 0.3782
# Epochs[1/3]--batch[200/938]--Acc: 0.9375--loss: 0.2796
# Epochs[1/3]--batch[250/938]--Acc: 0.9688--loss: 0.1311
# Epochs[1/3]--batch[300/938]--Acc: 1.0--loss: 0.0481
# Epochs[1/3]--batch[350/938]--Acc: 0.9531--loss: 0.1806
# Epochs[1/3]--batch[400/938]--Acc: 0.9375--loss: 0.2402
# Epochs[1/3]--batch[450/938]--Acc: 0.9219--loss: 0.3639
# Epochs[1/3]--batch[500/938]--Acc: 0.9844--loss: 0.1803
# Epochs[1/3]--batch[550/938]--Acc: 0.9844--loss: 0.1121
# Epochs[1/3]--batch[600/938]--Acc: 0.9219--loss: 0.2243
# Epochs[1/3]--batch[650/938]--Acc: 1.0--loss: 0.0281
# Epochs[1/3]--batch[700/938]--Acc: 0.9375--loss: 0.156
# Epochs[1/3]--batch[750/938]--Acc: 0.9688--loss: 0.1356
# Epochs[1/3]--batch[800/938]--Acc: 0.9688--loss: 0.1081
# Epochs[1/3]--batch[850/938]--Acc: 0.8906--loss: 0.2203
# Epochs[1/3]--batch[900/938]--Acc: 0.9844--loss: 0.0622
# Epochs[1/3]--Acc on test 0.9726
# Epochs[2/3]--batch[0/938]--Acc: 0.9844--loss: 0.0375
# Epochs[2/3]--batch[50/938]--Acc: 0.9688--loss: 0.0667
# Epochs[2/3]--batch[100/938]--Acc: 0.9531--loss: 0.075
# Epochs[2/3]--batch[150/938]--Acc: 0.9688--loss: 0.1261
# Epochs[2/3]--batch[200/938]--Acc: 0.9688--loss: 0.1541
# Epochs[2/3]--batch[250/938]--Acc: 0.9531--loss: 0.107
# Epochs[2/3]--batch[300/938]--Acc: 1.0--loss: 0.0129
# Epochs[2/3]--batch[350/938]--Acc: 0.9688--loss: 0.1385
# Epochs[2/3]--batch[400/938]--Acc: 1.0--loss: 0.0082
# Epochs[2/3]--batch[450/938]--Acc: 1.0--loss: 0.0341
# Epochs[2/3]--batch[500/938]--Acc: 1.0--loss: 0.013
# Epochs[2/3]--batch[550/938]--Acc: 0.9688--loss: 0.0342
# Epochs[2/3]--batch[600/938]--Acc: 0.9688--loss: 0.0374
# Epochs[2/3]--batch[650/938]--Acc: 0.9844--loss: 0.0785
# Epochs[2/3]--batch[700/938]--Acc: 1.0--loss: 0.0217
# Epochs[2/3]--batch[750/938]--Acc: 0.9688--loss: 0.0633
# Epochs[2/3]--batch[800/938]--Acc: 1.0--loss: 0.0194
# Epochs[2/3]--batch[850/938]--Acc: 0.9844--loss: 0.0591
# Epochs[2/3]--batch[900/938]--Acc: 0.9688--loss: 0.046
# Epochs[2/3]--Acc on test 0.9784
# Epochs[3/3]--batch[0/938]--Acc: 0.9688--loss: 0.0855
# Epochs[3/3]--batch[50/938]--Acc: 0.9844--loss: 0.0472
# Epochs[3/3]--batch[100/938]--Acc: 1.0--loss: 0.0244
# Epochs[3/3]--batch[150/938]--Acc: 0.9844--loss: 0.0274
# Epochs[3/3]--batch[200/938]--Acc: 0.9844--loss: 0.0252
# Epochs[3/3]--batch[250/938]--Acc: 1.0--loss: 0.0518
# Epochs[3/3]--batch[300/938]--Acc: 0.9844--loss: 0.0373
# Epochs[3/3]--batch[350/938]--Acc: 1.0--loss: 0.0083
# Epochs[3/3]--batch[400/938]--Acc: 1.0--loss: 0.0111
# Epochs[3/3]--batch[450/938]--Acc: 1.0--loss: 0.0175
# Epochs[3/3]--batch[500/938]--Acc: 0.9844--loss: 0.0612
# Epochs[3/3]--batch[550/938]--Acc: 0.9844--loss: 0.0335
# Epochs[3/3]--batch[600/938]--Acc: 1.0--loss: 0.0233
# Epochs[3/3]--batch[650/938]--Acc: 0.9688--loss: 0.0896
# Epochs[3/3]--batch[700/938]--Acc: 0.9844--loss: 0.0674
# Epochs[3/3]--batch[750/938]--Acc: 0.9844--loss: 0.0656
# Epochs[3/3]--batch[800/938]--Acc: 1.0--loss: 0.0121
# Epochs[3/3]--batch[850/938]--Acc: 0.9688--loss: 0.0797
# Epochs[3/3]--batch[900/938]--Acc: 1.0--loss: 0.0307
# Epochs[3/3]--Acc on test 0.9812
# 真实标签为：tensor([7, 2, 1, 0, 4])
# 预测标签为：tensor([7, 2, 1, 0, 4])
