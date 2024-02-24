"""
文件名: Code/Chapter09/C08_TextCNNAtt/TextRNN.py
创建时间: 2024/2/24 8:29 下午
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    """
    Luong's multiplicative attention
    """

    def __init__(self, hidden_size, dropout=0.):
        super(LuongAttention, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, key, value, src_key_padding_mask=None):
        """
        :param query:  global query : [1, hidden_size]
        :param key:    encoder_output [batch_size, src_len, hidden_size]
        :param value:  encoder_output [batch_size, src_len, hidden_size]
        :param src_key_padding_mask:  填充值标志True表示是填充值
        :return:
        """
        scores = torch.matmul(key, self.linear(query).transpose(0, 1))
        # [1, hidden_size] @ [hidden_size, hidden_size] = [1, hidden_size] ==> [hidden_size,1]
        # [batch_size, src_len, hidden_size] @  [hidden_size, 1]= [batch_size, src_len, 1]
        scores = scores.squeeze(-1)  # [batch_size, tgt_len]
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask, float('-inf'))
            # 掩盖掉填充部分的注意力值，[batch_size, tgt_len]
        attention_weights = torch.softmax(scores, dim=-1)  # [batch_size, src_len]
        context_vec = torch.bmm(self.drop(attention_weights).unsqueeze(1), value)
        # [batch_size, 1, src_len] @  [batch_size, src_len, hidden_size] = [batch_size, 1, hidden_size]
        return context_vec, attention_weights


class TextRNNAtt(nn.Module):
    def __init__(self, config):
        """
        :param num_classes: 分类数
        :param vocab_size: 指代的是词表的长度
        :param embedding_size: 指词向量的维度
        :param hidden_size:
        :param num_layers:
        :param cell_type: 'RNN'、'LSTM' 'GRU'
        :param bidirectional: False or True
        :param cat_type: 特征组合方式
        """
        super(TextRNNAtt, self).__init__()
        if config.cell_type == 'RNN':
            rnn_cell = nn.RNN
        elif config.cell_type == 'LSTM':
            rnn_cell = nn.LSTM
        elif config.cell_type == 'GRU':
            rnn_cell = nn.GRU
        else:
            raise ValueError("Unrecognized RNN cell type: " + config.cell_type)
        out_hidden_size = config.hidden_size * (int(config.bidirectional) + 1)
        self.config = config
        if config.cat_type == 'attention':
            self.global_query = nn.Parameter(torch.randn((1, out_hidden_size)))
            self.attention = LuongAttention(out_hidden_size)
        self.token_embedding = nn.Embedding(config.top_k, config.embedding_size)
        self.rnn = rnn_cell(config.embedding_size, config.hidden_size, num_layers=config.num_layers,
                            batch_first=True, bidirectional=config.bidirectional)
        self.classifier = nn.Sequential(nn.LayerNorm(out_hidden_size),
                                        nn.Linear(out_hidden_size, out_hidden_size),
                                        nn.ReLU(inplace=True), nn.Dropout(0.5),
                                        nn.Linear(out_hidden_size, config.num_classes))

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, src_len]
        :param labels: [batch_size, tgt_len]
        :return: logits: [batch_size, src_len, vocab_size]
        """
        x = self.token_embedding(x)  # [batch_size, src_len, embedding_size]
        x, _ = self.rnn(x)  # [batch_size, src_len, out_hidden_size]

        if self.config.cat_type == 'last':
            x = x[:, -1]  # [batch_size, out_hidden_size]
        elif self.config.cat_type == 'mean':
            x = torch.mean(x, dim=1)  # [batch_size, out_hidden_size]
        elif self.config.cat_type == 'sum':
            x = torch.sum(x, dim=1)  # [batch_size, out_hidden_size]
        elif self.config.cat_type == 'attention':
            x, atten_weights = self.attention(self.global_query, x, x)
            x = x.squeeze(1)  # [batch_size, 1, out_hidden_size] ==> [batch_size, out_hidden_size]
        else:
            raise ValueError("Unrecognized cat_type: " + self.cat_type)
        logits = self.classifier(x)  # [batch_size, num_classes]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


class ModelConfig(object):
    def __init__(self):
        self.num_classes = 15
        self.top_k = 8
        self.embedding_size = 16
        self.hidden_size = 64
        self.num_layers = 2
        self.cell_type = 'LSTM'
        self.bidirectional = True
        self.cat_type = 'attention'


if __name__ == '__main__':
    config = ModelConfig()
    model = TextRNNAtt(config)
    x = torch.randint(0, config.top_k, [2, 3], dtype=torch.long)
    label = torch.randint(0, config.num_classes, [2], dtype=torch.long)
    loss, logits = model(x, label)
    print(loss)
    print(logits)

    # inference
    out = model(torch.tensor([[5, 2, 6, 7, 7]]))
    print(out)
    print(out.shape)
