import sys

sys.path.append('../')
from model.Embedding import TokenEmbedding, PositionalEncoding
import torch

if __name__ == '__main__':
    x = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]], dtype=torch.long)
    x = x.reshape(5, 2)  # [src_len, batch_size]
    token_embedding = TokenEmbedding(vocab_size=11, emb_size=512)
    x = token_embedding(tokens=x)
    pos_embedding = PositionalEncoding(d_model=512)
    x = pos_embedding(x=x)
    print(x.shape)
