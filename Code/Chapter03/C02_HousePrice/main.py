"""
文件名: Code/Chapter03/C01_HousePrice/house_price.py
创建时间: 2023/1/7 2:48 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def make_house_data():
    """
    构造数据集
    :return:  x:shape [100,1] y:shape [100,1]
    """
    np.random.seed(20)
    x = np.random.randn(100, 1) + 5  # 面积
    noise = np.random.randn(100, 1)
    y = x * 2.8 - noise  # 价格
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def visualization(x, y, y_pred=None):
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.xlabel('面积', fontsize=15)
    plt.ylabel('房价', fontsize=15)
    plt.scatter(x, y, c='black')
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.plot(x, y_pred)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.tight_layout()  # 调整子图间距
    plt.show()


def train(x, y):
    epochs = 40
    lr = 0.003
    input_node = x.shape[1]
    output_node = 1
    net = nn.Sequential(nn.Linear(input_node, output_node))
    loss = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器
    for epoch in range(epochs):
        logits = net(x)
        l = loss(logits, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()  # 执行梯度下降
        print("Epoch: {}, loss: {}".format(epoch, l))
    logits = net(x)
    l = loss(logits, y)
    print("RMSE: {}".format(torch.sqrt(l / 2)))
    return logits.detach().numpy()


if __name__ == '__main__':
    x, y = make_house_data()
    y_pred = train(x, y)
    visualization(x, y, y_pred)
