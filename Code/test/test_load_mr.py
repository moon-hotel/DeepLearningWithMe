import sys

sys.path.append('../')

from utils import MR

if __name__ == '__main__':
    dataloader = MR(top_k=2000,
                    max_sen_len=None,
                    batch_size=4,
                    is_sample_shuffle=True,
                    cut_words=False)
    train_iter, val_iter = dataloader.load_train_val_test_data(is_train=True)
    for x,y in train_iter:
        print(x.shape)
        print(x)
        print(y)
        print(y.shape)
        break
    vocab = dataloader.get_vocab()
    print(vocab)

