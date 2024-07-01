from sklearn.datasets import make_classification
import torch
from torch.distributions import Dirichlet

x, y = make_classification(
    n_samples=10000,
    n_features=10,
    n_informative=10 // 2 + 1,
    n_redundant=1,
    n_classes=10,
    n_clusters_per_class=1,
)
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)
print(torch.unique(y, return_counts=True))
print("原始特征维度：", x.shape)


def dirichlet_split(train_labels, alpha, n_clients):
    n_classes = (train_labels.max() + 1).to(int)
    label_distribution = Dirichlet(
        torch.full((n_clients,), alpha, dtype=torch.float32)
    ).sample((n_classes,))
    class_idcs = [torch.nonzero(train_labels == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        total_size = len(c)
        splits = (fracs * total_size).int()
        splits[-1] = total_size - splits[:-1].sum()
        idcs = torch.split(c, splits.tolist())
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]

    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    return client_idcs


client_idcs = dirichlet_split(y, 1, 2)
print(client_idcs)

print(torch.unique(y[client_idcs[0]], return_counts=True))
print(torch.unique(y[client_idcs[1]], return_counts=True))
