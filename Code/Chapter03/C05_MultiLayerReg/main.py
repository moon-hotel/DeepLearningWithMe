"""
文件名: Code/Chapter03/C05_MultiLayerReg/main.py
创建时间: 2023/1/8 10:02 上午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
import numpy as np
import matplotlib.pyplot as plt


def make_trapezoid_data():
    np.random.seed(20)
    x1 = np.random.uniform(0.5, 1.5, [50, 1])
    x2 = np.random.uniform(0.5, 1.5, [50, 1])
    x = np.hstack((x1, x2))
    # 在这里我们便得到了一个50行2列的样本数据，其中第一列为上底，第二列为下底
    y = 0.5 * (x1 + x2) * x1
    return x, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_grad(z):
    """
    sigmoid梯度
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))


def loss(y, y_hat):
    """
    计算均方误差
    :param y: shape: [m,output_node]
    :param y_hat: shape: [m,output_node]
    :return: shape: [output_node]
    """
    y_hat = y_hat.reshape(y.shape)  # 一定要注意将两者reshape一样，不然容易出错且不易察觉
    return 0.5 * np.mean((y - y_hat) ** 2)


def forward(x, w1, b1, w2, b2):  # 预测
    """
    梯形面积预测前向传播
    :param x: shape: [m,input_node]
    :param w1: shape: [input_node,hidden_node]
    :param b1: shape: [hidden_node]
    :param w2: shape: [hidden_node,output_node]
    :param b2: shape: [output_node]
    :return:
    """
    a1 = x
    z2 = np.matmul(a1, w1) + b1
    a2 = sigmoid(z2)
    z3 = np.matmul(a2, w2) + b2
    return z3, a2


def gradient_descent(grads, params, lr):
    """
    执行梯度下降
    :param grads:
    :param params:
    :param lr:
    :return:
    """
    for i in range(len(grads)):
        params[i] -= lr * grads[i]
    return params


def backward(a3, w2, a2, a1, y):
    """
    反向传播计算梯度
    :param a3: shape: [m,output_node]
    :param w2: shape: [hidden_node,output_node]
    :param a2: shape: [m,hidden_node]
    :param a1: shape: [m,input_node]
    :param y:  shape: [m,output_node]
    :return:
    """
    m = a3.shape[0]  # 获取样本个数
    delta3 = (a3 - y) * 1.  # [m,output_node]
    grad_w2 = (1 / m) * np.matmul(a2.T, delta3)
    # [hidden_node,m] @ [m,output_node] = [hidden_node,output_node]
    grad_b2 = (1 / m) * np.sum(delta3, axis=0)
    # [m,output_node]==> [output_node]

    delta2 = np.matmul(delta3, w2.T) * sigmoid_grad(a2)
    # [m,output_node] @ [1,hidden_node] * [m,hidden_node] = [m,hidden_node]
    grad_w1 = (1 / m) * np.matmul(a1.T, delta2)
    # [input_node,m] @ [m,hidden_node] = [input_node,hidden_node]
    grad_b1 = (1 / m) * np.sum(delta2, axis=0)
    # [m,hidden_node] ==> [hidden_node]
    return [grad_w2, grad_b2, grad_w1, grad_b1]


def train(x, y):
    epochs = 1600
    lr = 0.08
    input_node = 2
    hidden_node = 80
    output_node = 1
    losses = []

    w1 = np.random.random([input_node, hidden_node])
    b1 = np.zeros(hidden_node)
    w2 = np.random.random([hidden_node, output_node])
    b2 = np.zeros(output_node)
    for i in range(epochs):
        logits, a2 = forward(x, w1, b1, w2, b2)
        l = loss(y, logits)
        grads = backward(logits, w2, a2, x, y)
        w2, b2, w1, b1 = gradient_descent(grads, [w2, b2, w1, b1], lr=lr)
        if i % 10 == 0:
            print("Epoch: {}, loss: {}".format(i, l))
        losses.append(l)
    logits, _ = forward(x, w1, b1, w2, b2)
    l = loss(logits, y)
    print("RMSE: {}".format(np.sqrt(l / 2)))
    print("真实值：", y[:5].reshape(-1))
    print("预测值：", logits[:5].reshape(-1))
    return losses, w1, b1, w2, b2

# Epoch: 0, loss: 408.4199890827408
# Epoch: 10, loss: 0.7597133963035992
# Epoch: 20, loss: 0.2798011448669906
# Epoch: 30, loss: 0.23229650438832639
# Epoch: 40, loss: 0.22791828168755696
# ......
# Epoch: 1570, loss: 0.00021502542755396308
# Epoch: 1580, loss: 0.00021467026015166345
# Epoch: 1590, loss: 0.00021435692503766647
# RMSE: 0.010346032331625699
# 真实值： [1.26355453 1.61181353 1.85784564 1.7236208  0.48818497]
# 预测值： [1.25302678 1.60291594 1.85990525 1.72523891 0.50386205]


def prediction(x, w1, b1, w2, b2):
    x = np.reshape(x, [-1, 2])
    logits, _ = forward(x, w1, b1, w2, b2)
    print(f"预测结果为：\n{logits}")
    # 预测结果为：
    # [[0.40299857]
    #  [0.82788597]]
    return logits


def visualization_loss(losses):
    plt.plot(range(len(losses)), losses)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(-.05, 0.5)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    x, y = make_trapezoid_data()
    losses, w1, b1, w2, b2 = train(x, y)
    visualization_loss(losses)
    x = np.array([[0.6, 0.8],
                  [0.7, 1.5]])
    prediction(x, w1, b1, w2, b2)
