from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np


def load_data():
    data = load_boston()
    x, y = data.data, data.target
    ss = StandardScaler()
    x = ss.fit_transform(x)  # 特征标准化
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def forward(x, weights, bias):  # 预测
    y = torch.matmul(x, weights) + bias
    return y


def loss(y, y_hat):
    return 0.5 * torch.mean((y - y_hat.reshape(y.shape)) ** 2)  # 一定要注意将两者reshape一样


def RMSE(y, y_hat):
    return torch.sqrt(loss(y, y_hat))


def gradientDescent(params, lr):
    for param in params:
        param.data -= lr * param.grad
        param.grad.zero_()


def train(x, y):
    epoches = 100
    lr = 0.3
    w = torch.tensor(np.random.normal(0, 0.1, [x.shape[1], 1]), dtype=torch.float32, requires_grad=True)
    b = torch.tensor(np.random.randn(1), dtype=torch.float32, requires_grad=True)
    for i in range(epoches):
        logits = forward(x, w, b)
        l = loss(y, logits)
        l.backward()
        gradientDescent([w, b], lr)
        if i % 10 == 0:
            print("Epoch: {}, loss: {}".format(i, l))
    logits = forward(x, w, b)
    print("RMSE: {}".format(RMSE(y, logits)))
    print("真实房价：", y[12])
    print("预测房价：", logits[12])


if __name__ == '__main__':
    x, y = load_data()
    train(x, y)
