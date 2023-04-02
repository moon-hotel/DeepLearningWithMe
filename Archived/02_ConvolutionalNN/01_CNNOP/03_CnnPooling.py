import torch.nn.functional as F
import torch

feature_maps = torch.tensor(
    [5, 2, 0, 1, 0, 0, 2, 3, 0, 7, 2, 3, 2,
     2, 1, 1, 6, 4, 8, 1, 2, 7, 1, 5, 9,
     4, 2, 0, 1, 0, 2, 7, 1, 3, 6, 2, 4, 2,
     2, 3, 2, 6, 9, 8, 0, 10, 7, 2, 5, 7],dtype=torch.float32).reshape([1, 2, 5, 5])
print(feature_maps)
result = F.max_pool2d(input=feature_maps, kernel_size=3, stride=1)
print(result)
