import torch
import numpy as np

array = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

target = torch.tensor([[1], [2]])
a = torch.tensor(array)
print(a.gather(1, target))