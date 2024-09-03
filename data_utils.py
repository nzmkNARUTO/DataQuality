import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm, trange
from scipy.stats import logistic
from copy import deepcopy
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from torch.distributions import Dirichlet

from model import train_model

import sys

sys.dont_write_bytecode = True


class VolumeDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def slice(self, indices):
        return VolumeDataset(self.x[indices], self.y[indices])

    def distribute(self):
        distribution = torch.bincount(self.y)
        print(distribution)


def generate_synthetic_classification_data(size=10000, dimension=10, classes=10):
    x, y = make_classification(
        n_samples=size,
        n_features=dimension,
        n_informative=int(dimension**0.5) + 1,
        n_redundant=0,
        n_classes=classes,
        n_clusters_per_class=1,
    )
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.int64)
    return VolumeDataset(x, y)


def dirichlet_split(dataset, n_clients, alpha):
    n_classes = (dataset.y.max() + 1).to(int)
    label_distribution = Dirichlet(
        torch.full((n_clients,), alpha, dtype=torch.float32)
    ).sample((n_classes,))
    class_idcs = [torch.nonzero(dataset.y == y).flatten() for y in range(n_classes)]
    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        total_size = len(c)
        splits = (fracs * total_size).int()
        splits[-1] = total_size - splits[:-1].sum()
        idcs = torch.split(c, splits.tolist())
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]

    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    datasets = []
    for i in range(n_clients):
        datasets.append(
            VolumeDataset(dataset.x[client_idcs[i]], dataset.y[client_idcs[i]])
        )
    return datasets


def even_split(dataset, n_clients):
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    client_size = len(dataset) // n_clients
    client_idcs = [
        indices[i * client_size : (i + 1) * client_size] for i in range(n_clients)
    ]
    datasets = []
    for i in range(n_clients):
        datasets.append(
            VolumeDataset(dataset.x[client_idcs[i]], dataset.y[client_idcs[i]])
        )
    return datasets


def get_mnist_datasets(n_participants, train):
    xs = []
    datasets = []
    for _ in range(n_participants):
        dataset = torchvision.datasets.MNIST(
            "./data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        xs.append(dataset.data.float())
        datasets.append(dataset)
    return xs, datasets


def generate_synthetic_data(d=1, size=50):
    data = torch.rand((size, d)) * (1 - 0) + 0
    # dataset = np.random.uniform(0, 1, (size, d))
    # dataset = np.random.normal(0, 1, (size,d))
    return data


def generate_linear_label(xTrain, d=1, weight=None, bias=None):
    weightBias = torch.cat((weight, bias))
    onePaddedX = torch.cat((xTrain, torch.ones((len(xTrain), 1))), axis=1)
    label = (onePaddedX @ weightBias).reshape(-1, 1)
    return label


def generate_weight_bias(d=1):
    weight = torch.normal(0, 1, size=(d,))
    bias = torch.normal(0, 1, size=(1,))
    return weight, bias


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


def regression2classification(yTrue: torch.Tensor, parameter: float) -> torch.Tensor:
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
    mean, std = yTrue.mean(0), yTrue.std(0)
    yTrue = (yTrue - mean) / std  # normalize
    yTrue = logistic.cdf(parameter * yTrue)
    y = (np.random.random(yTrue.shape[-1]) < yTrue).astype(np.float32)
    return torch.from_numpy(y)


def plot_figure(
    values: dict,
    xTrain: torch.Tensor,
    yTrain: torch.Tensor,
    xTest: torch.Tensor,
    yTest: torch.Tensor,
    baseModel: torch.nn.Module,
    loss_func: torch.nn.Module,
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
        _calculate_portion_performance(
            indexes=np.argsort(valueSource)[::-1],
            plotPoints=pointsPlot,
            xTrain=xTrain,
            yTrain=yTrain,
            xTest=xTest,
            yTest=yTest,
            baseModel=baseModel,
            loss_func=loss_func,
            metric=metric,
        )
        for valueSource in tqdm(valueSources, desc="Calculating performance")
    ]
    random = np.mean(
        [
            _calculate_portion_performance(
                indexes=np.random.permutation(
                    np.argsort(valueSources[0])[::-1]
                ),  # random exclude 1 datum
                plotPoints=pointsPlot,
                xTrain=xTrain,
                yTrain=yTrain,
                xTest=xTest,
                yTest=yTest,
                baseModel=baseModel,
                loss_func=loss_func,
                metric=metric,
            )
            for _ in trange(10, desc="Calculating random performance")
        ],
        0,
    )
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i in range(len(valuesKeys)):
        plt.plot(
            pointsPlot / len(xTrain) * 100,
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
        pointsPlot / len(xTrain) * 100,
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


def _calculate_portion_performance(
    indexes: list,
    plotPoints: list,
    xTrain: torch.Tensor,
    yTrain: torch.Tensor,
    xTest: torch.Tensor,
    yTest: torch.Tensor,
    baseModel: torch.nn.Module,
    loss_func: torch.nn.Module,
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
        torch.max(torch.bincount(yTest.squeeze(-1).int()).float() / len(yTest))
        .detach()
        .cpu()
    )
    for i in trange(
        len(plotPoints), 0, -1, desc="Calculating portion performance", leave=False
    ):
        keepIndexes = np.array(
            [idx for idx in indexes[plotPoints[i - 1] :]]
        )  # every plotPoints data
        x, y = xTrain[keepIndexes], yTrain[keepIndexes]
        if len(torch.unique(y)) == len(
            torch.unique(yTest)
        ):  # if select data contains all classes
            model = train_model(
                baseModel=deepcopy(baseModel),
                x=x,
                y=y,
                loss_func=loss_func,
                tqdm=True,
            )
            y_pred = model(xTest)
            scores.append(metric(y_pred, yTest).detach().cpu())
        else:
            scores.append(initScore)
    return np.array(scores)[::-1]
