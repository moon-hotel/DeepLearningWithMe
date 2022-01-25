import torch.nn as nn
import torch
import os
from copy import deepcopy


def para_state_dict(model, model_save_dir):
    state_dict = deepcopy(model.state_dict())
    model_save_path = os.path.join(model_save_dir, 'model.pt')
    if os.path.exists(model_save_path):
        loaded_paras = torch.load(model_save_path)
        for key in state_dict:  # 在新的网络模型中遍历对应参数
            if key in loaded_paras and state_dict[key].size() == loaded_paras[key].size():
                print("成功初始化参数:", key)
                state_dict[key] = loaded_paras[key]
    return state_dict


class LeNet6(nn.Module):
    def __init__(self, ):
        super(LeNet6, self).__init__()
        self.conv = nn.Sequential(  # [n,1,28,28]
            nn.Conv2d(1, 6, 5, padding=2),  # in_channels, out_channels, kernel_size
            nn.ReLU(),  # [n,6,24,24]
            nn.MaxPool2d(2, 2),  # kernel_size, stride  [n,6,14,14]
            nn.Conv2d(6, 16, 5),  # [n,16,10,10]
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # [n,16,5,5]
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, img, labels=None):
        output = self.conv(img)
        logits = self.fc(output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    model_save_path = os.path.join('./MODEL', 'model.pt')
    loaded_paras = torch.load(model_save_path)
    for param_tensor in loaded_paras:
        print(param_tensor, "\t", loaded_paras[param_tensor].size())

    print("Model paras in LeNet6:")
    model = LeNet6()
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
