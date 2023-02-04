"""
文件名: Code/Chapter03/C13_RegMetircs.py
创建时间: 2023/2/4 5:30 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np


def MAE(y, y_pre):
    return np.mean(np.abs(y - y_pre))


def MSE(y, y_pre):
    return np.mean((y - y_pre) ** 2)


def RMSE(y, y_pre):
    return np.sqrt(MSE(y, y_pre))


def MAPE(y, y_pre):
    return np.mean(np.abs((y - y_pre) / y))


def R2(y, y_pre):
    u = np.sum((y - y_pre) ** 2)
    v = np.sum((y - np.mean(y)) ** 2)
    return - (u / v)


np.random.seed(10)
if __name__ == '__main__':
    y_true = 2 * np.random.randn(200) + 1
    y_pred = np.random.randn(200) + y_true
    print(f"MAE: {MAE(y_true, y_pred)}\n"
          f"MSE: {MSE(y_true, y_pred)}\n"
          f"RMSE: {RMSE(y_true, y_pred)}\n"
          f"MAPE: {MAPE(y_true, y_pred)}\n"
          f"R2: {R2(y_true, y_pred)}\n")

# MAE: 0.7395229164418393
# MSE: 0.8560928033277224
# RMSE: 0.9252528321100792
# MAPE: 2.2088106952308864
# R2: -0.2245663206367467
