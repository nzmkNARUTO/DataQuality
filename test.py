import torch
from torchmetrics.classification import Accuracy

metric = Accuracy("binary")
x = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
y = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])

print(metric(x, y))
