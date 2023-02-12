"""
文件名: Code/Chapter03/C16_L2Regularization/main.py
创建时间: 2023/2/8 8:18 下午
"""
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt


def make_data():
    np.random.seed(1)
    n_train, n_test, n_features = 80, 110, 150
    w, b = np.random.randn(n_features, 1) * 0.01, 0.01
    x = np.random.normal(size=(n_train + n_test, n_features))
    y = np.matmul(x, w) + b
    y += np.random.normal(scale=0.3, size=y.shape)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    x_train, x_test = x[:n_train, :], x[n_train:, :]
    y_train, y_test = y[:n_train, :], y[n_train:, :]
    return x_train, x_test, y_train, y_test


def train(x_train, x_test, y_train, y_test, lambda_term=0.):
    torch.random.manual_seed(100)
    epochs = 300
    lr = 0.005
    input_node = x_train.shape[1]
    output_node = 1
    net = nn.Sequential(nn.Linear(input_node, output_node))
    loss = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=lambda_term)  # 定义优化器
    loss_train = []
    loss_test = []
    for epoch in range(epochs):
        logits = net(x_train)
        l = loss(logits, y_train)
        loss_train.append(l.item())
        logits = net(x_test)
        ll = loss(logits, y_test)
        loss_test.append(ll.item())
        optimizer.zero_grad()
        l.backward()
        optimizer.step()  # 执行梯度下降
    return loss_train, loss_test


def visualization(loss_train_1, loss_test_1, lambda_term_1,
                  loss_train_2, loss_test_2, lambda_term_2):
    iterations = len(loss_train_1)
    plt.figure(figsize=(8, 4))
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.subplot(1, 2, 1)
    plt.title(f"lambda = {lambda_term_1}", fontsize=15)
    plt.plot(range(iterations), loss_train_1, label='训练误差', c='black')
    plt.plot(range(iterations), loss_test_1, label='测试误差', linestyle='--', c='black')
    plt.legend(fontsize=13)
    plt.ylim(-0.05, .45)
    plt.xlabel('迭代次数', fontsize=15)

    plt.subplot(1, 2, 2)
    plt.title(f"lambda = {lambda_term_2}", fontsize=15)
    plt.plot(range(iterations), loss_train_2, label='训练误差', c='black')
    plt.plot(range(iterations), loss_test_2, label='测试误差', linestyle='--', c='black')
    plt.legend(fontsize=13)
    plt.ylim(-0.05, .45)
    plt.xlabel('迭代次数', fontsize=15)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = make_data()
    lambda_term_1 = 0.0
    lambda_term_2 = 5
    loss_train_1, loss_test_1 = train(x_train, x_test, y_train, y_test, lambda_term=lambda_term_1)
    loss_train_2, loss_test_2 = train(x_train, x_test, y_train, y_test, lambda_term=lambda_term_2)
    visualization(loss_train_1, loss_test_1, lambda_term_1,
                  loss_train_2, loss_test_2, lambda_term_2)
