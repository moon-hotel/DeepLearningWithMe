"""
文件名: Code/Chapter06/C01_LearningRate/mian.py
创建时间: 2023/4/3 7:52 下午
"""

import torch.nn as nn
import torch
from transformers import optimization
import matplotlib.pyplot as plt


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(2, 5)

    def forward(self, x):
        out = self.fc(x).sum()
        return out


if __name__ == '__main__':
    x = torch.rand([3, 2])
    model = Model()
    model.train()
    steps = 1000
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)

    # 1. constant
    # scheduler = optimization.get_constant_schedule(optimizer, last_epoch=-1)
    # name = "constant"

    # 2. constant_with_warmup
    # scheduler = optimization.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=300)
    # name = "constant_with_warmup"

    # 3. linear
    # scheduler = optimization.get_linear_schedule_with_warmup(optimizer,
    #                                 num_warmup_steps=300, num_training_steps=steps)
    # name = "linear"

    # 4. polynomial
    # scheduler = optimization.get_polynomial_decay_schedule_with_warmup(optimizer, num_warmup_steps=300,
    #                                                                    num_training_steps=steps,
    #                                                                    lr_end=1e-7, power=3)
    # name = "polynomial"

    # 5. cosine
    # scheduler = optimization.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=300,
    #                                                          num_training_steps=steps, num_cycles=2)
    # name = "cosine"

    # 6. cosine_with_restarts
    # scheduler = optimization.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
    #                                                                             num_warmup_steps=300,
    #                                                                             num_training_steps=steps, num_cycles=2)
    name = "cosine_with_restarts"

    scheduler = optimization.get_scheduler(name="cosine_with_restarts", optimizer=optimizer,
                              num_warmup_steps = 300, num_training_steps = steps)
    lrs = []
    for _ in range(steps):
        loss = model(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr())
    plt.figure(figsize=(7, 4))
    plt.plot(range(steps), lrs, label=name)
    plt.legend(fontsize=13)
    plt.show()