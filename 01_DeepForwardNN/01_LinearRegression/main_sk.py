from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np


def load_data():
    data = load_boston()
    x, y = data.data, data.target
    ss = StandardScaler()
    x = ss.fit_transform(x)  # 特征标准化
    return x, y


def RMSE(y, y_hat):
    mse = 0.5 * np.mean((y - y_hat.reshape(y.shape)) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def train(x, y):
    model = Ridge(alpha=0.)
    model.fit(x, y)
    y_pred = model.predict(x)
    print("RMSE: {}".format(RMSE(y, y_pred)))


if __name__ == '__main__':
    x, y = load_data()
    train(x, y)
