"""
文件名: Code/Chapter03/C12_MultiLayerCla/main.py
创建时间: 2023/2/2 8:33 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest

"""

import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def load_dataset():
    """
    构造数据集
    :return: x: shape (60000, 784)
             y: shape (60000, )
    """
    data = MNIST(root='~/Datasets/MNIST', download=True,
                 transform=transforms.ToTensor())
    x, y = [], []
    for img in data:
        x.append(np.array(img[0]).reshape(1, -1))
        y.append(img[1])
    x = np.vstack(x)
    y = np.array(y)
    return x, y


def gen_batch(x, y, batch_size=64):
    """
    构建迭代器
    :param x: 
    :param y: 
    :param batch_size: 
    :return: 
    """
    s_index, e_index, batches = 0, 0 + batch_size, len(y) // batch_size
    if batches * batch_size < len(y):
        batches += 1
    for i in range(batches):
        if e_index > len(y):
            e_index = len(y)
        batch_x = x[s_index:e_index]
        batch_y = y[s_index: e_index]
        s_index, e_index = e_index, e_index + batch_size
        yield batch_x, batch_y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def crossEntropy(y_true, logits):
    """
    计算交叉熵
    :param y_true:  one-hot [m,n]
    :param logits:  one-hot [m,n]
    :return:
    """
    loss = y_true * np.log(logits)
    return -np.sum(loss) / len(y_true)


def softmax(x):
    """
    :param x: [m,n]
    :return: [m,n]
    """
    s = np.exp(x)
    # 因为np.sum(s, axis=1)操作后变量的维度会减一，为了保证广播机制正常
    # 所以设置 keepdims=True 保持维度不变
    return s / np.sum(s, axis=1, keepdims=True)


def forward(x, w1, b1, w2, b2, w3, b3):
    """
    前向传播
    :param x: shape: [m,input_node]
    :param w1: shape: [input_node,hidden_node]
    :param b1: shape: [hidden_node]
    :param w2: shape: [hidden_node,output_node]
    :param b2: shape: [output_node]
    :return:
    """
    z2 = np.matmul(x, w1) + b1  # [m,n] @ [n,h] + [h] = [m,h]
    a2 = sigmoid(z2)  # a2:[m,h]
    z3 = np.matmul(a2, w2) + b2
    a3 = sigmoid(z3)
    z4 = np.matmul(a3, w3) + b3  # w2: [h,c]
    a4 = softmax(z4)
    return a4, a3, a2


def backward(a4, a3, a2, a1, w3, w2, y):
    m = a4.shape[0]
    delta4 = a4 - y  # [m,c]
    grad_w3 = 1 / m * np.matmul(a3.T, delta4)  # [hidden_nodes,c]
    grad_b3 = 1 / m * np.sum(delta4, axis=0)  # [c]

    delta3 = np.matmul(delta4, w3.T) * (a3 * (1 - a3))  # [m,hidden_nodes]
    grad_w2 = 1 / m * np.matmul(a2.T, delta3)  # [hidden_nodes,hidden_nodes]
    grad_b2 = 1 / m * (np.sum(delta3, axis=0))  # [hidden_nodes,]

    delta2 = np.matmul(delta3, w2.T) * (a2 * (1 - a2))  # [m,hidden_nodes]
    grad_w1 = 1 / m * np.matmul(a1.T, delta2)  # [input_nodes, hidden_nodes]
    grad_b1 = 1 / m * (np.sum(delta2, axis=0))  # [hidden_nodes,]
    return [grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3]


def gradient_descent(grads, params, lr):
    """
    梯度下降
    :param grads:
    :param params:
    :param lr:
    :return:
    """
    for i in range(len(grads)):
        params[i] -= lr * grads[i]
    return params


def accuracy(y_true, logits):
    """
    用于计算单个batch中预测结果的准确率
    :param y_true:
    :param logits:
    :return:
    """
    acc = (logits.argmax(1) == y_true).mean()
    return acc


def evaluate(x, y, net, w1, b1, w2, b2, w3, b3):
    """
    用于计算整个数据集所有预测结果的准确率
    :param x: [m,n]
    :param y: [m,]
    :param net:
    :param w1: [m,h]
    :param b1: [h,]
    :param w2: [h,c]
    :param b2: [c,]
    :return:
    """
    acc_sum, n = 0.0, 0
    for x, y in gen_batch(x, y):
        logits, _, _ = net(x, w1, b1, w2, b2, w3, b3)
        acc_sum += (logits.argmax(1) == y).sum()
        n += len(y)
    return acc_sum / n


def visualization_loss(losses):
    import matplotlib.pyplot as plt
    plt.plot(range(len(losses)), losses)
    plt.xlabel('迭代次数', fontsize=15)
    plt.ylabel('损失值', fontsize=15)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    # plt.ylim(-.05, 0.5)
    plt.tight_layout()
    plt.show()


def train(x_data, y_data):
    input_nodes = 28 * 28
    hidden_nodes = 1024
    output_nodes = 10
    epochs = 2
    lr = 0.03
    losses = []
    batch_size = 64
    w1 = np.random.uniform(-0.3, 0.3, [input_nodes, hidden_nodes])
    b1 = np.zeros(hidden_nodes)

    w2 = np.random.uniform(-0.3, 0.3, [hidden_nodes, hidden_nodes])
    b2 = np.zeros(hidden_nodes)

    w3 = np.random.uniform(-0.3, 0.3, [hidden_nodes, output_nodes])
    b3 = np.zeros(output_nodes)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(gen_batch(x_data, y_data, batch_size)):
            logits, a3, a2 = forward(x, w1, b1, w2, b2, w3, b3)
            y_one_hot = np.eye(output_nodes)[y]
            loss = crossEntropy(y_one_hot, logits)
            grads = backward(logits, a3, a2, x, w3, w2, y_one_hot)
            w1, b1, w2, b2, w3, b3 = gradient_descent(grads,
                                                      [w1, b1, w2, b2, w3, b3], lr)
            if i % 5 == 0:
                acc = accuracy(y, logits)
                print(f"Epochs[{epoch + 1}/{epochs}]--batch[{i}/{len(x_data) // batch_size}]"
                      f"--Acc: {round(acc, 4)}--loss: {round(loss, 4)}")
            losses.append(loss)
    acc = evaluate(x_data, y_data, forward, w1, b1, w2, b2, w3, b3)
    print(f"Acc: {acc}")
    return losses, w1, b1, w2, b2, w3, b3


def prediction(x, w1, b1, w2, b2, w3, b3):
    x = x.reshape(-1, 784)
    logits, _, _ = forward(x, w1, b1, w2, b2, w3, b3)
    return np.argmax(logits, axis=1)


#
#
if __name__ == '__main__':
    x, y = load_dataset()
    losses, w1, b1, w2, b2, w3, b3 = train(x, y)
    visualization_loss(losses)
    y_pred = prediction(x[0], w1, b1, w2, b2, w3, b3)
    print(f"预测标签为: {y_pred}, 真实标签为: {y[0]}")

    # Epochs[1/2]--batch[0/937]--Acc: 0.1406--loss: 5.1525
    # Epochs[1/2]--batch[5/937]--Acc: 0.1562--loss: 2.5282
    # Epochs[1/2]--batch[10/937]--Acc: 0.2188--loss: 2.3137
    # Epochs[1/2]--batch[15/937]--Acc: 0.2812--loss: 2.0799
    # Epochs[1/2]--batch[20/937]--Acc: 0.5469--loss: 1.5729
    # Epochs[1/2]--batch[25/937]--Acc: 0.5--loss: 1.5751
    # ......
    # Epochs[2/2]--batch[930/937]--Acc: 0.9844--loss: 0.0769
    # Epochs[2/2]--batch[935/937]--Acc: 1.0--loss: 0.0262
    # Acc: 0.9115333333333333
