import sys

sys.path.append('../')
from model.MyTransformer import MyTransformer
import torch

if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    dmodel = 32
    tgt_len = 6
    num_head = 8
    src = torch.rand((src_len, batch_size, dmodel))  # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [batch_size, src_len]

    tgt = torch.rand((tgt_len, batch_size, dmodel))  # shape: [tgt_len, batch_size, embed_dim]
    tgt_key_padding_mask = torch.tensor([[True, True, True, False, False, False],
                                         [True, True, True, True, False, False]])  # shape: [batch_size, tgt_len]

    #   ============ 测试 MyMultiheadAttention ============
    # my_mh = MyMultiheadAttention(embed_dim=dmodel, num_heads=num_head)
    # r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)

    #  ============ 测试 MyTransformerEncoderLayer ============
    # my_transformer_encoder_layer = MyTransformerEncoderLayer(d_model=dmodel, nhead=num_head)
    # r = my_transformer_encoder_layer(src=src, src_key_padding_mask=src_key_padding_mask)

    #  ============ 测试 MyTransformerDecoder ============
    # my_transformer_encoder_layer = MyTransformerEncoderLayer(d_model=dmodel, nhead=num_head)
    # my_transformer_encoder = MyTransformerEncoder(encoder_layer=my_transformer_encoder_layer,
    #                                               num_layers=2,
    #                                               norm=nn.LayerNorm(dmodel))
    # memory = my_transformer_encoder(src=src, mask=None, src_key_padding_mask=src_key_padding_mask)
    # print(memory.shape)

    #
    # my_transformer_decoder_layer = MyTransformerDecoderLayer(d_model=dmodel, nhead=num_head)
    # my_transformer_decoder = MyTransformerDecoder(decoder_layer=my_transformer_decoder_layer,
    #                                               num_layers=1,
    #                                               norm=nn.LayerNorm(dmodel))
    # out = my_transformer_decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=tgt_key_padding_mask,
    #                              memory_key_padding_mask=src_key_padding_mask)
    # print(out.shape)

    # ============ 测试 MyTransformer ============
    my_transformer = MyTransformer(d_model=dmodel, nhead=num_head, num_encoder_layers=6,
                                   num_decoder_layers=6, dim_feedforward=500)
    src_mask = my_transformer.generate_square_subsequent_mask(src_len)
    tgt_mask = my_transformer.generate_square_subsequent_mask(tgt_len)
    out = my_transformer(src=src, tgt=tgt, tgt_mask=tgt_mask,
                         src_key_padding_mask=src_key_padding_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=src_key_padding_mask)
    print(out.shape)
