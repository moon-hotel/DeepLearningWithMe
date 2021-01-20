import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_feature_maps(use_bn=True, epochs=20):
    feature_map1 = np.load(f"feature_maps1_{use_bn}_epoch{epochs}.npy")
    feature_map2 = np.load(f"feature_maps2_{use_bn}_epoch{epochs}.npy")
    feature_map3 = np.load(f"feature_maps3_{use_bn}_epoch{epochs}.npy")
    feature_map4 = np.load(f"feature_maps4_{use_bn}_epoch{epochs}.npy")
    feature_maps = [feature_map1, feature_map2, feature_map3, feature_map4]
    plt.figure(figsize=(20, 5))
    plt.title(f"use_bn = {use_bn}")
    x, y = normal_distribution(mu=0, sigma2=1., n=100)
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.plot(x, y,c='black')
        sns.distplot(feature_maps[i],color='#a5c8e1')
        plt.xlabel(f"feature map {i + 1}")
    plt.tight_layout()
    plt.show()


def normal_distribution(mu=0, sigma2=1., n=100):
    x = np.linspace(-5, 5, 2 * n)
    y = (1 / np.sqrt(2 * np.pi) * sigma2) * np.exp(-((x - mu) ** 2) / (2 * sigma2 ** 2))
    return x, y


if __name__ == '__main__':
    plot_feature_maps(use_bn=True, epochs=19)
