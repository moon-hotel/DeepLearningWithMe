import sys

sys.path.append('../')

from utils import MR4ELMo

if __name__ == '__main__':
    dataloader = MR4ELMo(batch_size=4,
                         is_sample_shuffle=True)
    train_iter, val_iter = dataloader.load_train_val_test_data(is_train=True)
    for x, y in train_iter:
        print(x.shape)
        print(x)
        print(y)
        print(y.shape)
        break
