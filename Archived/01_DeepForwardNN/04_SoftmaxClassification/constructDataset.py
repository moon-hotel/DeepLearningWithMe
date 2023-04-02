import torch


def make_dataset():
    x = torch.linspace(0, 100, 100, dtype=torch.float32).reshape(-1, 2)
    y = torch.randn(50)
    dataset = torch.utils.data.TensorDataset(x, y)
    return dataset


if __name__ == '__main__':
    dataset = make_dataset()
    train_iter = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    for x, y in train_iter:
        print(x.shape)


