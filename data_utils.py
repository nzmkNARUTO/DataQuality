import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm, trange
from scipy.stats import logistic
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import trainModel

import sys

sys.dont_write_bytecode = True


def get_mnist_datasets(n_participants):
    trainDatasets = []
    testDatasets = []
    trainDatasets.append(
        DataLoader(
            dataset=datasets.MNIST(
                "../data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=64,
            shuffle=True,
        )
    )

    testDatasets.append(
        DataLoader(
            dataset=datasets.MNIST(
                "../data",
                train=False,
                transform=transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                ),
            ),
            batch_size=64,
            shuffle=True,
        )
    )
    return trainDatasets, testDatasets


def get_synthetic_datasets(n_participants, d=1, sizes=[], s=50, ranges=None):
    """
    Args:
        n_participants (int): number of data subsets to generate
        d (int): dimension
        sizes (list of int): number of data samples for each participant, if supplied
        s (int): number of data samples for each participant (equal), if supplied
        ranges (list of list): the lower and upper bound of the input domain for each participant, if supplied

    Returns:
        list containing the generated synthetic datasets for all participants
    """

    if 0 == len(sizes):
        sizes = torch.ones(n_participants, dtype=int) * s

    datasets = []
    for i, size in enumerate(sizes):
        if ranges != None:
            dataset = (
                torch.rand((size, d)) * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
            )
        else:
            dataset = torch.rand((size, d)) * (1 - 0) + 0
            # dataset = np.random.uniform(0, 1, (size, d))
            # dataset = np.random.normal(0, 1, (size,d))
        datasets.append(dataset.reshape(-1, d))
    return datasets


def generate_linear_labels(x_train, d=1, weights=None, bias=None):

    # generate random true weights and the bias
    if weights is None:
        weights = torch.normal(0, 1, size=(d,))
    if bias is None:
        bias = torch.normal(0, 1, size=(1,))

    labels = []
    weight_bias = torch.cat((weights, bias))
    for X in x_train:
        one_padded_X = torch.cat((X, torch.ones((len(X), 1))), axis=1)
        y = (one_padded_X @ weight_bias).reshape(-1, 1)
        labels.append(y)
    return labels, weights, bias


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
