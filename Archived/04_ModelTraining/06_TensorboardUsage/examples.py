from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import torch
import numpy as np


def add_graph(writer):
    img = torch.rand([1, 3, 64, 64], dtype=torch.float32)
    model = torchvision.models.AlexNet(num_classes=10)
    writer.add_graph(model, input_to_model=img)  # 类似于TensorFlow 1.x 中的fed


def add_scalar(writer):
    for n_iter in range(100):
        writer.add_scalar(tag='Loss/train',
                          scalar_value=np.random.random(),
                          global_step=n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)


def add_scalars(writer):
    r = 5
    for i in range(100):
        writer.add_scalars(main_tag='scalars1/P1',
                           tag_scalar_dict={'xsinx': i * np.sin(i / r),
                                            'xcosx': i * np.cos(i / r),
                                            'tanx': np.tan(i / r)},
                           global_step=i)
        writer.add_scalars('scalars1/P2',
                           {'xsinx': i * np.sin(i / (2 * r)),
                            'xcosx': i * np.cos(i / (2 * r)),
                            'tanx': np.tan(i / (2 * r))}, i)
        writer.add_scalars(main_tag='scalars2/Q1',
                           tag_scalar_dict={'xsinx': i * np.sin((2 * i) / r),
                                            'xcosx': i * np.cos((2 * i) / r),
                                            'tanx': np.tan((2 * i) / r)},
                           global_step=i)
        writer.add_scalars('scalars2/Q2',
                           {'xsinx': i * np.sin(i / (0.5 * r)),
                            'xcosx': i * np.cos(i / (0.5 * r)),
                            'tanx': np.tan(i / (0.5 * r))}, i)


def add_histogram(writer):
    for i in range(10):
        x = np.random.random(1000)
        writer.add_histogram('distribution centers/p1', x + i, i)
        writer.add_histogram('distribution centers/p2', x + i * 2, i)


def add_image(writer):
    from PIL import Image
    img1 = np.random.randn(1, 100, 100)
    writer.add_image('img/imag1', img1)
    img2 = np.random.randn(100, 100, 3)
    writer.add_image('img/imag2', img2, dataformats='HWC')
    img = Image.open('dufu.png')
    img_array = np.array(img)
    writer.add_image('local/dufu', img_array, dataformats='HWC')


def add_images(writer):
    img1 = np.random.randn(8, 100, 100, 1)
    writer.add_images('imgs/imags1', img1, dataformats='NHWC')
    img2 = np.zeros((16, 3, 100, 100))
    for i in range(16):
        img2[i, 0] = np.arange(0, 10000).reshape(100, 100) / 10000 / 16 * i
        img2[i, 1] = (1 - np.arange(0, 10000).reshape(100, 100) / 10000) / 16 * i
    writer.add_images('imgs/imags2', img2)  # Default is :math:`(N, 3, H, W)`


def add_figure(writer):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes([0.12, 0.1, 0.85, 0.8])
    "The dimensions [left, bottom, width, height] of the new axes."
    xx = np.arange(-5, 5, 0.01)
    ax.plot(xx, np.sin(xx), label="sin(x)")
    ax.legend()
    fig.suptitle('Sin(x) figure\n\n', fontweight="bold")
    writer.add_figure("figure", fig, 4)


def add_figures(writer, images, labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    labels = [text_labels[int(i)] for i in labels]
    fit, ax = plt.subplots(len(images) // 5, 5, figsize=(10, 2 * len(images) // 5))
    for i, axi in enumerate(ax.flat):
        image, label = images[i].reshape([28, 28]).numpy(), labels[i]
        axi.imshow(image)
        axi.set_title(label)
        axi.set(xticks=[], yticks=[])
    writer.add_figure("figures", fit)
    # plt.tight_layout()
    # plt.show()


def add_pr_curve(writer):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import label_binarize
    def get_dataset():
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        x, y = load_iris(return_X_y=True)
        random_state = np.random.RandomState(2020)
        n_samples, n_features = x.shape
        # 为数据增加噪音维度以便更好观察pr曲线
        x = np.concatenate([x, random_state.randn(n_samples, 100 * n_features)], axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5,
                                                            random_state=random_state)
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = get_dataset()
    model = LogisticRegression(multi_class="ovr")
    model.fit(x_train, y_train)
    y_scores = model.predict_proba(x_test)  # shape: (n,3)
    b_y = label_binarize(y_test, classes=[0, 1, 2])  # shape: (n,3)
    for i in range(3):
        writer.add_pr_curve(f"pr_curve/label_{i}", b_y[:, i], y_scores[:, i], global_step=1)


def add_embedding(writer):
    import tensorflow as tf
    import tensorboard as tb
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
    import keyword
    import torch
    # 随机生成100个标签信息
    meta = []
    while len(meta) < 100:
        meta = meta + keyword.kwlist  # get some strings
    meta = meta[:100]
    for i, v in enumerate(meta):
        meta[i] = v + str(i)
    # 随机生成100个标签图片
    label_img = torch.rand(100, 3, 10, 32)
    for i in range(100):
        label_img[i] *= i / 100.0
    data_points = torch.randn(100, 5)  # 随机生成100个点
    writer.add_embedding(mat=data_points, metadata=meta, label_img=label_img, global_step=1)


def load_dataset():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    return mnist_train, mnist_test


if __name__ == '__main__':
    w = SummaryWriter("runs/result_1")
    add_graph(w)
    add_scalar(w)
    add_scalars(w)
    add_histogram(w)
    add_image(w)
    add_images(w)
    add_figure(w)

    _, mnist_test = load_dataset()
    images = [mnist_test[i][0] for i in range(10)]
    labels = [mnist_test[i][1] for i in range(10)]
    add_figures(w, images, labels)

    add_pr_curve(w)
    add_embedding(w)
    w.close()
