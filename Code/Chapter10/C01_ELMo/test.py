from ELMo import ELMoCharacterCNN
from ELMo import ELMoBiLSTM
from ELMo import ELMo
import torch


class ModelConfig(object):
    def __init__(self):
        super().__init__()
        self.char_embed_num = 262
        self.char_embed_dim = 16
        self.max_characters_per_token = 50
        self.char_cnn_filters = [[1, 32], [2, 32], [3, 64],
                                 [4, 128], [5, 256], [6, 512], [7, 1024]]
        self.n_filters = sum(f[1] for f in self.char_cnn_filters) # 2048
        self.projection_dim = 512
        self.n_layers = 2


def test_ELMoCharacterCNN():
    config = ModelConfig()
    model = ELMoCharacterCNN(config)
    token_ids = torch.randint(0, 100, [2, 6, 50])
    print(model(token_ids).shape)


def test_ELMoBiLSTM():
    config = ModelConfig()
    token_embedding = torch.rand([2, 6, config.projection_dim])
    model = ELMoBiLSTM(config)
    for i in range(config.n_layers + 1):
        print("layer: ", i, model(token_embedding)[i].shape)


def test_ELMo():
    config = ModelConfig()
    token_ids = torch.randint(0, 100, [2, 6, 50])
    model = ELMo(config)
    outputs, elmo_rep = model(token_ids)
    print(outputs.shape)  # [n_layers+1, batch_size, seq_len, projection_dim*2]
    print(elmo_rep.shape)  # [batch_size, seq_len, projection_dim]


if __name__ == '__main__':
    # test_ELMoCharacterCNN()
    # test_ELMoBiLSTM()
    test_ELMo()
