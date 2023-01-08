"""
文件名: Code/Chapter03/C02_Trapezoid/main.py
创建时间: 2023/1/7 6:16 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def make_trapezoid_data():
    np.random.seed(20)
    x1 = np.random.randint(5, 10, [50, 1]) / 10
    x2 = np.random.randint(10, 16, [50, 1]) / 10
    x = np.hstack((x1, x2))
    # 在这里我们便得到了一个50行2列的样本数据，其中第一列为上底，第二列为下底
    y = 0.5 * (x1 + x2) * x1
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def visualization_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def train(x, y):
    epochs = 1000
    lr = 0.003
    input_node = x.shape[1]
    hidden_node = 50
    output_node = 1
    losses = []
    net = nn.Sequential(nn.Linear(input_node, hidden_node),
                        nn.Sigmoid(),
                        nn.Linear(hidden_node, output_node))
    loss = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 定义优化器
    for epoch in range(epochs):
        logits = net(x)
        l = loss(logits, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()  # 执行梯度下降
        print("Epoch: {}, loss: {}".format(epoch, l))
        losses.append(l.item())
    logits = net(x)
    l = loss(logits, y)
    print("RMSE: {}".format(torch.sqrt(l / 2)))
    print("真实值：", y[:5].detach().numpy().reshape(-1))
    print("预测值：", logits[:5].detach().numpy().reshape(-1))
    return losses


# Epoch: 0, loss: 0.21947528421878815
# Epoch: 1, loss: 0.15144017338752747
# Epoch: 2, loss: 0.09904950112104416
# Epoch: 3, loss: 0.062260374426841736
# Epoch: 4, loss: 0.04036729037761688
# Epoch: 5, loss: 0.03167763724923134
# Epoch: 6, loss: 0.033266790211200714
# Epoch: 7, loss: 0.04110710322856903
# Epoch: 8, loss: 0.05081081762909889
# ......
# Epoch: 998, loss: 0.00018733780598267913
# Epoch: 999, loss: 0.0001872947468655184
# RMSE: 0.009676053188741207
# 真实值： [0.84  0.7   0.855 0.665 0.54 ]
# 预测值： [0.83861655 0.7117218  0.85532755 0.6754495  0.54401684]

if __name__ == '__main__':
    x, y = make_trapezoid_data()
    losses = train(x, y)
    visualization_loss(losses)
