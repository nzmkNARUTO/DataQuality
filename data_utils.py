import torch
from scipy.stats import logistic


def f(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, POLY_DEGREE: int
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

    Returns:
    --------
    return: torch.Tensor
        the output data
    """
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
    return w * features + b


def regression2classification(y_true: torch.Tensor) -> torch.Tensor:
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
    return (y_true < 0.5).float()
