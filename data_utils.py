import torch
import numpy as np
from scipy.stats import logistic

import sys

sys.dont_write_bytecode = True


def f(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, POLY_DEGREE: int, IMPORTANT: int
) -> torch.Tensor:
    """
    The function to be learned

    Parameters:
    -----------
    x: torch.Tensor
        the input data
    w: torch.Tensor
        the weight
    b: torch.Tensor
        the bias
    POLY_DEGREE: int
        the degree of the polynomial
    IMPORTANT: int
        the number of important dimensions

    Returns:
    --------
    return: torch.Tensor
        the output data
    """
    importantDimension = np.random.choice(x.shape[-1], IMPORTANT, replace=False)
    x = x[:, importantDimension]
    features = []  # every degree of x
    n = x.shape[0]
    exp = x  # higher degree of x
    holder = x  # the contemporary degree of x
    for _ in range(1, POLY_DEGREE + 1):
        features.append(
            torch.reshape(holder, [n, -1])
        )  # add the contemporary degree of x
        exp = exp.unsqueeze(-1)  # move to higher degree
        holder = exp * holder.unsqueeze(1)  # calculate the higher degree of x
    features = torch.cat(features, -1)  # concatenate all the degrees of x
    return torch.sum(w * features + b, -1).unsqueeze(-1)


def regression2classification(y_true: torch.Tensor, parameter: float) -> torch.Tensor:
    """
    Convert regression label to classification binary label

    Parameters:
    -----------
    y_true: torch.Tensor
        the regression label

    Returns:
    --------
    return: torch.Tensor
        the classification binary label
    """
    mean, std = y_true.mean(0), y_true.std(0)
    y_true = (y_true - mean) / std  # normalize
    y_true = logistic.cdf(parameter * y_true)
    y = (np.random.random(y_true.shape[-1]) < y_true).astype(np.float32)
    return torch.from_numpy(y)
