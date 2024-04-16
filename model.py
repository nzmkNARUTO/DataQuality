import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm, trange
from copy import deepcopy


class LogisticRegressionModel(torch.nn.Module):

    def __init__(self, X_DIMENSION):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(X_DIMENSION, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        y_pred = F.sigmoid(y_pred)
        return y_pred


def trainModel(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lossFunction: torch.nn.Module,
    epochs: int = 5000,
    tqdm: bool = True,
) -> tuple[torch.nn.Module, torch.Tensor]:
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

    Returns:
    --------
    return: tuple[torch.nn.Module, Tensor]
        the trained [model, loss]
    """
    model = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

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
        if deltaLoss < 1e-4 or previousLoss < 1e-2:
            break
        loss = optimizer.step()
    if tqdm:
        t.close()
    return model
