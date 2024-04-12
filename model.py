import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from multiprocessing import cpu_count
from tqdm import tqdm, trange
from copy import deepcopy


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, POLY_DEGREE):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(POLY_DEGREE, 1)

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
    # model = deepcopy(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    previousLoss = (
        1e10  # previous loss, for calculating delta loss, if delta loss < 1e-10, break
    )
    with trange(epochs, desc="Training", leave=False) as t:
        for _ in t:
            y_pred = model(x)
            loss = lossFunction(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            t.set_postfix(loss=loss.item())
            deltaLoss = abs(previousLoss - loss.item())
            previousLoss = loss.item()
            if deltaLoss < 1e-3 or loss.item() < 1e-1:
                break
            loss = optimizer.step()
    return model


def _trainModelWithoutTqdm(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lossFunction: torch.nn.Module,
    epochs: int = 5000,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    previousLoss = (
        1e10  # previous loss, for calculating delta loss, if delta loss < 1e-10, break
    )
    for _ in range(epochs):
        y_pred = model(x)
        loss = lossFunction(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        deltaLoss = previousLoss - loss.item()
        previousLoss = loss.item()
        if deltaLoss < 1e-3 or loss.item() < 1e-1:
            break
        loss = optimizer.step()
    return model


def trainModelMultiprocess(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    lossFunction: torch.nn.Module,
    epochs: int = 5000,
) -> torch.nn.Module:
    model = deepcopy(model)
    mp.set_start_method("spawn", force=True)
    model.share_memory()
    processes = []
    cpuNumber = cpu_count()
    epochs = int(epochs / cpuNumber)
    pool = mp.Pool(cpuNumber)
    for _ in range(cpuNumber):
        p = pool.apply_async(
            _trainModelWithoutTqdm,
            args=(model, x, y, lossFunction, epochs),
        )
        processes.append(p)
    pool.close()
    pool.join()
    return model

    # for _ in range(cpuNumber):
    #     p = mp.Process(
    #         target=_trainModelWithoutTqdm,
    #         args=(model, x, y, lossFunction, epochs),
    #     )
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()
    # return model
