import torch
import torch.nn as nn
from allennlp.modules.token_embedders import ElmoTokenEmbedder


class ELMoClassification(nn.Module):
    def __init__(self, num_classes=10, freeze=True):
        super().__init__()
        self.elmo_rep = ElmoTokenEmbedder(requires_grad=not freeze)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x, labels=None):
        """
        :param x: [batch_size, src_len]
        :param labels: [batch_size]
        :return:
        """
        features = self.elmo_rep(x)[:, 0]
        # [batch_size, seq_len, 1024] ==> [batch_size, 1024]
        logits = self.classifier(features)  # [batch_size, num_classes]
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    token_ids = torch.randint(0, 100, [2, 6, 50])
    labels = torch.tensor([0, 1], dtype=torch.long)
    model = ELMoClassification(num_classes=2, freeze=False)
    loss, logits = model(token_ids, labels)
    print(logits.shape)
    print(loss)
    for (name, param) in model.named_parameters():
        print(param.requires_grad)
