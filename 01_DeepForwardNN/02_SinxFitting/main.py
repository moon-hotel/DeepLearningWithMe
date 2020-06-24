import torch
import numpy as np
import matplotlib.pyplot as plt


def make_data():
    num_samples = 500
    x = torch.linspace(-np.pi, np.pi, num_samples, dtype=torch.float32).reshape(-1, 1)
    y = torch.sin(x)+torch.tensor(np.random.normal(0,0.05,[num_samples,1]))
    return x, y


def loss(y, y_hat):
    return 0.5 * torch.mean((y - y_hat.reshape(y.shape)) ** 2)  # 一定要注意将两者reshape一样


def gradientDescent(params, lr):
    for param in params:
        param.data -= lr * param.grad
        param.grad.zero_()


def forward(x, w1, b1, w2, b2):
    out1 = torch.matmul(x, w1) + b1
    out1 = torch.sigmoid(out1)
    out2 = torch.matmul(out1, w2) + b2
    return out2


def train(x, y):
    input_nodes = x.shape[1]
    hidden_nodes = 50
    output_nodes = 1
    epoches = 8000
    lr = 0.1
    w1 = torch.tensor(np.random.normal(0, 0.2, [input_nodes, hidden_nodes]), dtype=torch.float32, requires_grad=True)
    b1 = torch.tensor(0, dtype=torch.float32, requires_grad=True)
    w2 = torch.tensor(np.random.normal(0, 0.2, [hidden_nodes, output_nodes]), dtype=torch.float32, requires_grad=True)
    b2 = torch.tensor(0, dtype=torch.float32, requires_grad=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.pause(6)
    ax.scatter(x, y,s=10)
    plt.ylim(-1, 1)
    plt.ion()
    plt.show()
    for i in range(epoches):
        logits = forward(x, w1, b1, w2, b2)
        l = loss(y, logits)
        l.backward()
        gradientDescent([w1, b1, w2, b2], lr)
        if i % 500 == 0:
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            lines = ax.plot(x, logits.detach().numpy())
            plt.pause(0.2)
            print("Epoch:{}, Loss:{}".format(i,l))
    logits = forward(x, w1, b1, w2, b2)
    rmse = torch.sqrt(loss(y, logits))
    print(rmse)


if __name__ == '__main__':
    x, y = make_data()
    train(x, y)
