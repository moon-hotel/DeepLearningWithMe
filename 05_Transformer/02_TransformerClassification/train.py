import torch.nn as nn
import torch
from config import Config
from data_helper import LoadSentenceClassificationDataset, my_tokenizer
from ClassificationModel import ClassificationModel
import os
import time
from copy import deepcopy


class CustomSchedule(nn.Module):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        self.step = 1.

    def __call__(self):
        arg1 = self.step ** -0.5
        arg2 = self.step * (self.warmup_steps ** -1.5)
        self.step += 1.
        return (self.d_model ** -0.5) * min(arg1, arg2)


def train_model(config):
    data_loader = LoadSentenceClassificationDataset(config.train_corpus_file_paths,
                                                    my_tokenizer,
                                                    batch_size=config.batch_size,
                                                    min_freq=config.min_freq,
                                                    max_sen_len=config.max_sen_len)
    train_iter, test_iter = data_loader.load_train_val_test_data(
        config.train_corpus_file_paths, config.test_corpus_file_paths)

    classification_model = ClassificationModel(vocab_size=len(data_loader.vocab),
                                               d_model=config.d_model,
                                               nhead=config.num_head,
                                               num_encoder_layers=config.num_encoder_layers,
                                               dim_feedforward=config.dim_feedforward,
                                               dim_classification=config.dim_classification,
                                               num_classification=config.num_class,
                                               dropout=config.dropout)

    for p in classification_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model_save_path = os.path.join(config.model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        classification_model.load_state_dict(loaded_paras)
        print("## 成功载入已有模型，进行追加训练......")
    classification_model = classification_model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss()
    learning_rate = CustomSchedule(config.d_model)
    optimizer = torch.optim.Adam(classification_model.parameters(),
                                 lr=0.,
                                 betas=(config.beta1, config.beta2),
                                 eps=config.epsilon)
    classification_model.train()
    max_test_acc = 0
    for epoch in range(config.epochs):
        losses = 0
        start_time = time.time()
        for idx, (sample, label) in enumerate(train_iter):
            sample = sample.to(config.device)  # [src_len, batch_size]
            label = label.to(config.device)
            padding_mask = (sample == data_loader.PAD_IDX).transpose(0, 1)
            logits = classification_model(sample,
                                          src_key_padding_mask=padding_mask)  # [batch_size,num_class]
            optimizer.zero_grad()
            loss = loss_fn(logits, label)
            loss.backward()
            lr = learning_rate()
            for p in optimizer.param_groups:
                p['lr'] = lr
            optimizer.step()
            losses += loss.item()

            acc = (logits.argmax(1) == label).float().mean()
            if idx % 10 == 0:
                print(f"Epoch: {epoch}, Batch[{idx}/{len(train_iter)}], "
                      f"Train loss :{loss.item():.3f}, Train acc: {acc:.3f}")
        end_time = time.time()
        train_loss = losses / len(train_iter)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if (epoch + 1) % config.model_save_per_epoch == 0:
            acc = evaluate(test_iter, classification_model, config.device)
            print(f"Accuracy on test {acc:.3f}, max acc on test {max_test_acc:.3f}")
            if acc > max_test_acc:
                max_test_acc = acc
                state_dict = deepcopy(classification_model.state_dict())
                torch.save(state_dict, model_save_path)


def evaluate(data_iter, model, device):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            acc_sum += (logits.argmax(1) == y).float().sum().item()
            n += len(y)
        model.train()
        return acc_sum / n


if __name__ == '__main__':
    config = Config()
    train_model(config)
    """
    Epoch: 9, Batch: [410/469], Train loss 0.186, Train acc: 0.938
    Epoch: 9, Batch: [420/469], Train loss 0.150, Train acc: 0.938
    Epoch: 9, Batch: [430/469], Train loss 0.269, Train acc: 0.941
    Epoch: 9, Batch: [440/469], Train loss 0.197, Train acc: 0.925
    Epoch: 9, Batch: [450/469], Train loss 0.245, Train acc: 0.917
    Epoch: 9, Batch: [460/469], Train loss 0.272, Train acc: 0.902
    Accuracy on test 0.886
    """
