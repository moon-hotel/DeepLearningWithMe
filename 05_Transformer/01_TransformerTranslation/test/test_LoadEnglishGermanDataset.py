import sys

sys.path.append('../')

from utils.data_helpers import LoadEnglishGermanDataset
from utils.data_helpers import my_tokenizer
from config.config import Config

if __name__ == '__main__':
    config = Config()
    data_loader = LoadEnglishGermanDataset(config.test_corpus_file_paths,
                                           batch_size=config.batch_size,
                                           tokenizer=my_tokenizer,
                                           min_freq=config.min_freq)
    train_iter, valid_iter, test_iter = data_loader.load_train_val_test_data(config.test_corpus_file_paths,
                                                                             config.test_corpus_file_paths,
                                                                             config.test_corpus_file_paths)
    print(data_loader.PAD_IDX)
    for src, tgt in train_iter:
        tgt_input = tgt[:-1, :]
        tgt_out = tgt[1:, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = data_loader.create_mask(src, tgt_input)
        print("src shape：", src.shape)  # [de_tensor_len,batch_size]
        print(src.transpose(0, 1)[:3])
        print("tgt shape:", tgt.shape)  # [de_tensor_len,batch_size]
        print("src input shape:", src.shape)
        print("src_padding_mask shape (batch_size, src_len): ", src_padding_mask.shape)
        print("tgt input shape:", tgt_input.shape) # [tgt_len,batch_size]
        print(tgt_input.transpose(0,1)[:3])
        print("tgt_padding_mask shape: (batch_size, tgt_len) ", tgt_padding_mask.shape)
        print("tgt output shape:", tgt_out.shape) # [tgt_len,batch_size]
        print(tgt_out.transpose(0,1)[:3])
        print("tgt_mask shape (tgt_len,tgt_len): ", tgt_mask.shape)
        break
