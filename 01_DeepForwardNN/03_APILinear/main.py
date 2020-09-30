from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


def load_data():
    data = load_boston()
    x, y = data.data, data.target
    ss = StandardScaler()
    x = ss.fit_transform(x)  # 特征标准化
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y


def train(x, y):
    epoches = 500
    lr = 0.08
    input_node = x.shape[1]
    hidden_nodes = [10, 20, 10]
    output_node = 1
    net = nn.Sequential(
        nn.Linear(input_node, hidden_nodes[0]),  # 第一层 shape: [1,10]
        nn.Sigmoid(),  # 非线性变换
        nn.Linear(hidden_nodes[0], hidden_nodes[1]),  # 第二层 shape: [10,20]
        nn.Sigmoid(),
        nn.Linear(hidden_nodes[1], hidden_nodes[2]),  # 第三层 shape: [20,10]
        nn.Sigmoid(),
        nn.Linear(hidden_nodes[2], output_node)  # 第四层 shape: [10,1]
    )

    loss = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 定义优化器
    for epoch in range(epoches):
        logits = net(x)
        l = loss(logits.reshape(y.shape), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()  # 执行梯度下降
        print("Epoch: {}, loss: {}".format(epoch, l))
    logits = net(x)
    l = loss(logits.reshape(y.shape), y)
    print("RMSE: {}".format(torch.sqrt(l / 2)))
    print("真实房价：", y[12])
    print("预测房价：", logits[12])


if __name__ == '__main__':
    x, y = load_data()
    train(x, y)
