import numpy as np
import torch
import torch.multiprocessing as mp
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanSquaredError
from copy import deepcopy
from tqdm import trange
from data_utils import f, regression2classification, plot_figure
from model import LogisticRegressionModel, train_model
from node import Client, Server

import sys

sys.dont_write_bytecode = True

if __name__ == "__main__":

    n_participants = M = 2
    TRAIN_SIZE = 50000
    TEST_SIZE = 10000
    D = 6
    CLIENT_NUMBER = 2
    mp.set_start_method("fork", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    baseModel = LogisticRegressionModel(D).to(device)
    metric = MeanSquaredError()
    loss_func = torch.nn.MSELoss()

    server = Server(
        baseModel=baseModel, metric=metric, dimension=D, size=TEST_SIZE, threshold=0.01
    )
    for i in range(CLIENT_NUMBER):
        client = Client(
            baseModel=deepcopy(baseModel),
            metric=metric,
            loss_func=loss_func,
            weight=server.weight,
            bias=server.bias,
            dimension=D,
            size=TRAIN_SIZE,
        )
        server.clients.append(client)

    server.train()
