import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from copy import deepcopy

import sys

sys.dont_write_bytecode = True


class LogisticRegressionModel(torch.nn.Module):

    def __init__(self, X_DIMENSION):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(X_DIMENSION, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        # y_pred = F.sigmoid(y_pred)
        return y_pred


class ClassificationModel(torch.nn.Module):

    def __init__(self, X_DIMENSION, Y_DIMENSION):
        super(ClassificationModel, self).__init__()
        self.linear = torch.nn.Linear(X_DIMENSION, 20)
        self.linear2 = torch.nn.Linear(20, Y_DIMENSION)

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = F.relu(y_pred)
        y_pred = self.linear2(y_pred)
        y_pred = F.softmax(y_pred, dim=1)
        return y_pred


def train_model(
    baseModel: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    loss_func: torch.nn.Module,
    batchSize: int = 32,
    epochs: int = 1000,
    tqdm: bool = True,
    learningRate: float = 0.1,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Module:
    """
    Train the model

    Parameters:
    -----------
    model: pytorch model
        the model to be trained
    x: Tensor
        the input data
    y: Tensor
        the target data
    criterion: torch.nn.Module
        the loss function
    tqdm: bool
        whether to use tqdm

    Returns:
    --------
    return: torch.nn.Module
        the trained model
    """
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True, num_workers=8)
    loss_func = loss_func.to(device)
    model = deepcopy(baseModel).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

    previousLoss = (
        1e10  # previous loss, for calculating delta loss, if delta loss < 1e-10, break
    )
    if tqdm:
        t = trange(epochs, desc="Training", leave=False)
    else:
        t = range(epochs)
    for _ in t:
        for idx, (x, y) in enumerate(dataloader):
            yPred = model(x.to(device))
            loss = loss_func(yPred, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            t.set_postfix(loss=f"{loss.item():.4f}")
            presentLoss = loss.item()
            loss = optimizer.step()
        deltaLoss = abs(previousLoss - presentLoss)
        previousLoss = presentLoss
        if previousLoss < 1e-2 or deltaLoss < 1e-8:
            break
    if tqdm:
        t.close()
    return model


def test_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    metric: torch.nn.Module,
    batchSize: int = 32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Test the model

    Parameters:
    -----------
    model: pytorch model
        the model to be tested
    x: Tensor
        the input data
    y: Tensor
        the target data
    metric: torch.nn.Module
        the metric to be used

    Returns:
    --------
    return: Tensor
        the metric result
    """
    dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
    metric = metric.to(device)
    model = model.to(device)
    metric.reset()
    for x, y in dataloader:
        yPred = model(x.to(device))
        metric(yPred, y.to(device))
    return metric.compute()
