import torch
import torch.nn as nn
from MyTransformer import MyTransformerEncoder, MyTransformerEncoderLayer
from Embedding import PositionalEncoding, TokenEmbedding


class ClassificationModel(nn.Module):
    def __init__(self, vocab_size=None,
                 d_model=512, nhead=8,
                 num_encoder_layers=6,
                 dim_feedforward=2048,
                 dim_classification=64,
                 num_classification=4,
                 dropout=0.1):
        super(ClassificationModel, self).__init__()
        self.pos_embedding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.src_token_embedding = TokenEmbedding(vocab_size, d_model)
        encoder_layer = MyTransformerEncoderLayer(d_model, nhead,
                                                  dim_feedforward,
                                                  dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = MyTransformerEncoder(encoder_layer,
                                            num_encoder_layers, encoder_norm)
        self.classifier = nn.Sequential(nn.Linear(d_model, dim_classification),
                                        nn.Dropout(dropout),
                                        nn.Linear(dim_classification, num_classification))

    def forward(self,
                src,  # [src_len, batch_size]
                src_mask=None,
                src_key_padding_mask=None,  # [batsh_size, src_len]
                concat_type='sum'  # 解码之后取所有位置相加，还是最后一个位置作为输出
                ):
        src_embed = self.src_token_embedding(src)  # [src_len, batch_size, embed_dim]
        src_embed = self.pos_embedding(src_embed)  # [src_len, batch_size, embed_dim]
        memory = self.encoder(src=src_embed,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)
        # [src_len,batch_size,embed_dim]
        if concat_type == 'sum':
            memory = torch.sum(memory, dim=0)
        elif concat_type == 'avg':
            memory = torch.sum(memory, dim=0) / memory.size(0)
        else:
            memory = memory[-1, ::]  # 取最后一个时刻
        # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        out = self.classifier(memory)  # 输出logits
        return out  # [batch_size, num_class]


if __name__ == '__main__':
    src_len = 7
    batch_size = 2
    dmodel = 32
    num_head = 4
    src = torch.tensor([[4, 3, 2, 6, 0, 0, 0],
                        [5, 7, 8, 2, 4, 0, 0]]).transpose(0, 1)  # 转换成 [src_len, batch_size]
    src_key_padding_mask = torch.tensor([[True, True, True, True, False, False, False],
                                         [True, True, True, True, True, False, False]])
    model = ClassificationModel(vocab_size=10, d_model=dmodel, nhead=num_head)
    logits = model(src, src_key_padding_mask=src_key_padding_mask)
    print(logits.shape)
