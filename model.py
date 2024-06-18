import torch
import torch.nn.functional as F
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


def trainModel(
    baseModel: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lossFunction: torch.nn.Module,
    epochs: int = 5000,
    tqdm: bool = True,
    learningRate: float = 0.1,
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
    model = deepcopy(baseModel)
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
        y_pred = model(x)
        loss = lossFunction(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        if tqdm:
            t.set_postfix(loss=f"{loss.item():.4f}")
        deltaLoss = abs(previousLoss - loss.item())
        previousLoss = loss.item()
        if previousLoss < 1e-2 or deltaLoss < 1e-8:
            break
        loss = optimizer.step()
    if tqdm:
        t.close()
    return model
