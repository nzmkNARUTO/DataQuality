import sys
import torch
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanSquaredError
from copy import deepcopy
from data_utils import (
    VolumeDataset,
    generate_synthetic_data,
    generate_linear_label,
    generate_weight_bias,
    generate_synthetic_classification_data,
    dirichlet_split,
    even_split,
)
from volume import Volume
from model import LogisticRegressionModel, ClassificationModel, train_model, test_model
from node import Node, Client, Server

import sys

sys.dont_write_bytecode = True

if __name__ == "__main__":
    TRAINSIZE = 500
    TESTSIZE = 100
    CLIENT_NUMBER = 2
    mp.set_start_method("fork", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    baseModel = ClassificationModel(5, 5)
    metric = Accuracy("multiclass", num_classes=5)
    loss_func = torch.nn.CrossEntropyLoss()

    dataset = generate_synthetic_classification_data(
        size=TRAINSIZE + TESTSIZE, dimension=5, classes=5
    )
    indices = list(range(TRAINSIZE + TESTSIZE))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:TRAINSIZE], indices[TRAINSIZE:]
    train_dataset, test_dataset = dataset.slice(train_indices), dataset.slice(
        test_indices
    )

    metrics = []
    # train_datasets = even_split(train_dataset, CLIENT_NUMBER)
    train_datasets = dirichlet_split(train_dataset, CLIENT_NUMBER, 0.8)
    print(torch.bincount(train_datasets[0].y))
    print(torch.bincount(train_datasets[1].y))
    server = Server(
        globalModel=deepcopy(baseModel),
        metric=metric,
        threshold=0.8,
        dataset=test_dataset,
        aggregation_method="fedavg",
        istopk=False,
        loss_func=loss_func,
    )
    for i in range(CLIENT_NUMBER):
        client = Client(
            globalModel=deepcopy(baseModel),
            metric=metric,
            loss_func=loss_func,
            dataset=train_datasets[i],
            epochs=5,
            istopk=True,
        )
        server.clients.append(client)

    metrics.append(server.train())

    server = Server(
        globalModel=deepcopy(baseModel),
        metric=metric,
        threshold=0.8,
        dataset=test_dataset,
        aggregation_method="fedtest",
        istopk=True,
        loss_func=loss_func,
    )
    for i in range(CLIENT_NUMBER):
        client = Client(
            globalModel=deepcopy(baseModel),
            metric=metric,
            loss_func=loss_func,
            dataset=train_datasets[i],
            epochs=5,
            istopk=True,
        )
        server.clients.append(client)

    metrics.append(server.train())
    plt.plot(metrics[0], label="fedavg")
    plt.plot(metrics[1], label="fedtest")
    plt.legend()
    plt.savefig("result.png")
