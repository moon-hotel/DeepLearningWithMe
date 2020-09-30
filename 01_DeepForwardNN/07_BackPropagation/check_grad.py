import numpy as np
import torch


def softmax(x):
    s = torch.exp(x)
    return s / torch.sum(s, dim=1, keepdim=True)


z = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.1, 0.1]], requires_grad=True)
a = softmax(z)
y = torch.tensor([[0, 0, 1], [1, 0, 0]])
l = -(y * torch.log(a)).sum()
l.backward()

grad_z = a - y
print(grad_z)
print(z.grad)
