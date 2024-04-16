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
    x1 = np.random.uniform(0.5, 1.5, [50, 1])
    x2 = np.random.uniform(0.5, 1.5, [50, 1])
    x = np.hstack((x1, x2))
    # 在这里我们便得到了一个50行2列的样本数据，其中第一列为上底，第二列为下底
    y = 0.5 * (x1 + x2) * x1
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def visualization_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.ylim(-0.1, 0.5)
    plt.xlabel('迭代次数',fontsize=15)
    plt.ylabel('损失值',fontsize=15)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.show()


def train(x, y):
    epochs = 1000
    lr = 0.125
    input_node = x.shape[1]
    hidden_node = 80
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
    print("预测结果：", net(torch.tensor([[0.6, 0.8], [0.7, 1.5]])))
    return losses


# Epoch: 1, loss: 21.451648712158203
# Epoch: 2, loss: 1.7119826078414917
# Epoch: 3, loss: 2.09726619720459
# Epoch: 4, loss: 6.766840934753418
# Epoch: 5, loss: 7.076871395111084
# Epoch: 6, loss: 4.739165306091309
# Epoch: 7, loss: 2.368345260620117
# ......
# Epoch: 997, loss: 5.002863326808438e-05
# Epoch: 998, loss: 4.9905607738764957e-05
# Epoch: 999, loss: 4.978356810170226e-05
# RMSE: 0.004983085673302412
# 真实值： [1.2635546  1.6118135  1.8578457  1.7236208  0.48818496]
# 预测值： [1.2622473 1.620144  1.855083  1.7280536 0.5039229]

if __name__ == '__main__':
    x, y = make_trapezoid_data()
    losses = train(x, y)
    visualization_loss(losses)
