import torch
import torch.nn.functional as F
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
    model = deepcopy(model)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_prev = (
        1e10  # previous loss, for calculating delta loss, if delta loss < 1e-10, break
    )
    with trange(epochs, desc="Training", leave=False) as t:
        for _ in t:

            def closure():  # closure function for optimizer.step in LBFGS
                y_pred = model(x)
                loss = lossFunction(y_pred, y)
                # print(f"Loss: {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                return loss

            loss = optimizer.step(closure=closure)
            t.set_postfix(loss=loss.item())
            delta_loss = loss_prev - loss.item()
            loss_prev = loss.item()
            if delta_loss < 1e-5:  # if delta loss < 1e-10, break
                break
            if loss.item() < 1e-1:  # if loss < 1e-1, break
                break

            # y_pred = model(x)
            # loss = criterion(y_pred, y)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # t.set_postfix(loss=loss.item())
    return model, loss
