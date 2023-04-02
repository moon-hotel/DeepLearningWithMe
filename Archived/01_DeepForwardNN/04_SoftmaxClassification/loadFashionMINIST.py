import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
import time


def load_dataset():
    mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True,
                                                    transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True,
                                                   transform=transforms.ToTensor())
    return mnist_train, mnist_test


def visualization(images, labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    labels = [text_labels[int(i)] for i in labels]
    fit, ax = plt.subplots(len(images) // 5, 5, figsize=(10, 2 * len(images) // 5))
    for i, axi in enumerate(ax.flat):
        image, label = images[i].reshape([28, 28]).numpy(), labels[i]
        axi.imshow(image)
        axi.set_title(label)
        axi.set(xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    mnist_train, mnist_test = load_dataset()
    image, label = mnist_train[0][0], mnist_train[0][1]
    print(image.shape, label)
    print(type(mnist_train))
    print(len(mnist_train), len(mnist_test))
    X, y = [], []
    for i in range(10):
        X.append(mnist_train[i][0])
        y.append(mnist_train[i][1])
    # visualization(X, y)

    start = time.time()
    train_iter = torch.utils.data.DataLoader(mnist_test, batch_size=1024, shuffle=True, num_workers=2)
    for x_test, y_test in train_iter:
        print(x_test.shape)
    print('%.2f' % (time.time() - start))
