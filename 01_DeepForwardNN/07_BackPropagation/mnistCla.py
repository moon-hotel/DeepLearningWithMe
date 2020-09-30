from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np


def load_dataset():
    """

    :return: x_train [m,n] y_train [m,]
    """
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = train_test_split(x / 255., y, test_size=0.3)
    return x_train, y_train, x_test, y_test


def gen_batch(x, y, batch_size=64):
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
    :param y_true:  one-hot [m,n]
    :param logits:  one-hot [m,n]
    :return:
    """
    return -np.sum(y_true * np.log(logits)) / len(y_true)


def softmax(x):
    """
    :param x: [m,n]
    :return: [m,n]
    """
    s = np.exp(x)
    return s / np.sum(s, axis=1, keepdims=True)


def forward_propagation(x, w1, b1, w2, b2):
    z2 = np.matmul(x, w1) + b1  # x:[m,n] w1:[n,h]
    a2 = sigmoid(z2)  # a2:[m,h]
    z3 = np.matmul(a2, w2) + b2  # w2: [h,c]
    a3 = softmax(z3)
    return a3, a2


def backward_propagation(a3, w2, a2, a1, y):
    m = a3.shape[0]
    delta3 = a3 - y  # [m,c]
    grad_w2 = 1 / m * np.matmul(a2.T, delta3)  # [h,m]@[m,c]=[h,c]
    delta2 = np.matmul(delta3, w2.T) * (a2 * (1 - a2))  # [m,h]
    grad_w1 = 1 / m * np.matmul(a1.T, delta2)  # [n,h]
    grad_b2 = 1 / m * (np.sum(delta3, axis=0))
    grad_b1 = 1 / m * (np.sum(delta2, axis=0))
    return grad_w2, grad_b2, grad_w1, grad_b1


def gradient_descent(grads, params, lr):
    for i in range(len(grads)):
        params[i] -= lr * grads[i]
    return params


def accuracy(y_true, logits):
    acc = (logits.argmax(1) == y_true).mean()
    return acc


def evaluate(x_test, y_test, net, w1, b1, w2, b2):
    acc_sum, n = 0.0, 0
    for x, y in gen_batch(x_test, y_test):
        logits, _ = net(x, w1, b1, w2, b2)
        acc_sum += (logits.argmax(1) == y).sum()
        n += len(y)
    return acc_sum / n


def train(x_train, y_train, x_test, y_test):
    input_nodes = 64
    hidden_nodes = 512
    output_nodes = 10
    epochs = 101
    lr = 1.
    batch_size = 64
    w1 = np.random.uniform(-0.3, 0.3, [input_nodes, hidden_nodes])
    b1 = np.zeros(hidden_nodes)
    w2 = np.random.uniform(-0.3, 0.3, [hidden_nodes, output_nodes])
    b2 = np.zeros(output_nodes)

    for epoch in range(epochs):
        for i, (x, y) in enumerate(gen_batch(x_train, y_train, batch_size)):
            logits, a2 = forward_propagation(x, w1, b1, w2, b2)
            y_one_hot = np.eye(output_nodes)[y]
            loss = crossEntropy(y_one_hot, logits)
            grad_w2, grad_b2, grad_w1, grad_b1 = backward_propagation(logits, w2, a2, x, y_one_hot)
            w2, b2, w1, b1 = gradient_descent([grad_w2, grad_b2, grad_w1, grad_b1],
                                              [w2, b2, w1, b1], lr)
            if i % 5 == 0:
                acc = accuracy(y, logits)
                print("Epochs[{}/{ }]---batch[{}/{}]---acc {:.4}---loss {:.4}".format(
                    epochs, epoch, len(x_train) // batch_size, i, acc, loss))

        if epoch % 10 == 0:
            acc = evaluate(x_test, y_test, forward_propagation, w1, b1, w2, b2)
            print("Epochs[{}/{}]--acc on test {:.4}".format(epochs, epoch, acc))


#
#
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()
    train(x_train, y_train, x_test, y_test)
