import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm, trange
from scipy.stats import logistic
from copy import deepcopy

from model import trainModel

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


def plotFigure(
    values: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    baseModel: torch.nn.Module,
    lossFunction: torch.nn.Module,
    metric: torch.nn.Module,
    numsOfPlotMarkers: int = 20,
) -> None:
    """
    Plot the result figure

    Parameters:
    -----------
    values: list
        the values need to be plotted
    numsOfPlotMarkers: int
        the number of plot markers

    """
    valuesKeys = list(values.keys())
    values = [values[key] for key in valuesKeys]
    plt.rcParams["figure.figsize"] = 8, 8
    plt.rcParams["font.size"] = 25
    plt.xlabel("Fraction of train data removed (%)", fontsize=20)
    plt.ylabel("Prediction accuracy (%)", fontsize=20)
    valueSources = [
        np.array([np.sum(value[i]) for i in range(len(value))]) for value in values
    ]
    if len(values[0]) < numsOfPlotMarkers:
        numsOfPlotMarkers = len(values[0]) - 1
    pointsPlot = np.arange(
        0,
        max(len(values[0]) - 10, numsOfPlotMarkers),
        max(len(values[0]) // numsOfPlotMarkers, 1),
    )
    performance = [
        _calculatePortionPerformance(
            indexes=np.argsort(valueSource)[::-1],
            plotPoints=pointsPlot,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            baseModel=baseModel,
            lossFunction=lossFunction,
            metric=metric,
        )
        for valueSource in tqdm(valueSources, desc="Calculating performance")
    ]
    random = np.mean(
        [
            _calculatePortionPerformance(
                indexes=np.random.permutation(
                    np.argsort(valueSources[0])[::-1]
                ),  # random exclude 1 datum
                plotPoints=pointsPlot,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                baseModel=baseModel,
                lossFunction=lossFunction,
                metric=metric,
            )
            for _ in trange(10, desc="Calculating random performance")
        ],
        0,
    )
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(len(valuesKeys)):
        plt.plot(
            pointsPlot / len(x_train) * 100,
            performance[i] * 100,
            "-",
            lw=5,
            ms=10,
            color=mcolors.TABLEAU_COLORS[colors[i]],
            label=valuesKeys[i],
        )
    # plt.plot(
    #     pointsPlot / len(x_train) * 100,
    #     performance[0] * 100,
    #     "-",
    #     lw=5,
    #     ms=10,
    #     color="b",
    #     label="a",
    # )
    # plt.plot(
    #     pointsPlot / len(x_train) * 100,
    #     performance[-1] * 100,
    #     "-.",
    #     lw=5,
    #     ms=10,
    #     color="g",
    #     label="LOO",
    # )
    plt.plot(
        pointsPlot / len(x_train) * 100,
        random * 100,
        "-",
        lw=5,
        ms=10,
        color=mcolors.TABLEAU_COLORS[colors[i + 1]],
        label="Random",
    )
    plt.legend()
    plt.savefig(f"result.png", bbox_inches="tight")
    plt.close()


def _calculatePortionPerformance(
    indexes: list,
    plotPoints: list,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    baseModel: torch.nn.Module,
    lossFunction: torch.nn.Module,
    metric: torch.nn.Module,
) -> np.array:
    """
    Calculate the portion performance

    Parameters:
    -----------
    indexes: list
        the indexes of the data
    plotPoints: list
        the plot points

    Returns:
    --------
    return: np.array
        the portion performance
    """
    scores = []
    initScore = (
        torch.max(torch.bincount(y_test.squeeze(-1).int()).float() / len(y_test))
        .detach()
        .cpu()
    )
    for i in trange(
        len(plotPoints), 0, -1, desc="Calculating portion performance", leave=False
    ):
        keepIndexes = np.array(
            [idx for idx in indexes[plotPoints[i - 1] :]]
        )  # every plotPoints data
        x, y = x_train[keepIndexes], y_train[keepIndexes]
        if len(torch.unique(y)) == len(
            torch.unique(y_test)
        ):  # if select data contains all classes
            model = trainModel(
                baseModel=deepcopy(baseModel),
                x=x,
                y=y,
                lossFunction=lossFunction,
                tqdm=True,
            )
            y_pred = model(x_test)
            scores.append(metric(y_pred, y_test).detach().cpu())
        else:
            scores.append(initScore)
    return np.array(scores)[::-1]
