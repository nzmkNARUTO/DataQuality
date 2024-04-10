import torch
import numpy as np
from torch.nn.modules import Module
from torchmetrics.metric import Metric
from copy import deepcopy
from tqdm import tqdm, trange
from model import trainModel


def looScore(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    baseModel: torch.nn.Module,
    lossFunction: torch.nn.Module,
    metric: Metric,
    baseR2Score: float,
) -> list[float]:
    """
    Calculate Leave-One-Out score

    Parameters:
    -----------
    x_train: torch.Tensor
        the input data of training dataset
    y_train: torch.Tensor
        the target data of training dataset
    x_test: torch.Tensor
        the input data of testing dataset
    y_test: torch.Tensor
        the target data of testing dataset
    baseModel: torch.nn.Module
        the base model
    lossFunction: torch.nn.Module
        the loss function
    metric: Metric
        the metric function
    baseR2Score: float
        the base R2 score

    Returns:
    --------
    return: list[float]
        the Leave-One-Out score
    """

    LOO_R2Score = []
    for i in trange(len(x_train), desc="Caculating LOO"):
        model, _ = trainModel(
            model=deepcopy(baseModel),
            x=torch.cat([x_train[:i], x_train[i + 1 :]]),
            y=torch.cat([y_train[:i], y_train[i + 1 :]]),
            lossFunction=lossFunction,
        )  # train model without i-th data
        y_pred = model(x_test)
        r2Score = metric(y_pred, y_test)
        LOO_R2Score.append((baseR2Score - r2Score).cpu().detach().numpy())
    return LOO_R2Score


class Shapley:

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        baseModel: torch.nn.Module,
        lossFunction: torch.nn.Module,
        metric: Metric,
        errorThreshold: float,
        truncatedRounds: int,
    ) -> None:
        self.x_train = x_train  # the input data of training dataset
        self.y_train = y_train  # the target data of training dataset
        self.x_test = x_test  # the input data of testing dataset
        self.y_test = y_test  # the target data of testing dataset
        self.baseModel = baseModel  # the base model
        self.lossFunction = lossFunction  # the loss function
        self.metric = metric  # the metric function, R2 score
        self.errorThreshold = errorThreshold  # the error threshold
        self.truncatedRounds = truncatedRounds  # truncate every truncatedRounds
        self.convergence = 0  # the convergence?
        self.memory = []
        self.indexes = []
        self.std = None
        self.mean = None

    def bagScore(self) -> tuple[float, float]:
        """
        Calculate Bagging mean&std

        Returns:
        --------
        return: tuple[float, float]
            the Bagging [std, mean]
        """
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
        baseModel = deepcopy(self.baseModel)
        criterion = self.lossFunction
        metric = self.metric

        model = trainModel(
            model=deepcopy(baseModel), x=x_train, y=y_train, lossFunction=criterion
        )
        bagScore = []
        for _ in trange(100, desc="Calculating Bagging std&mean"):
            bagIndexes = np.random.choice(
                len(y_test), len(y_test)
            )  # random extraction with replacement
            y_pred = model(x_test[bagIndexes])
            r2Score = metric(y_pred, y_test[bagIndexes])
            bagScore.append(r2Score.unsqueeze(0))
        self.std, self.mean = torch.std_mean(torch.cat(bagScore, dim=0))

    def calculateError(self):
        memory = self.memory
        if len(memory) < self.truncatedRounds:
            return 1.0
        lastRoundValue = np.cumsum(memory, axis=0) / np.reshape(
            np.arange(1, len(memory) + 1), (-1, 1)
        )
        errors = np.mean(
            np.abs(lastRoundValue[-self.truncatedRounds :] - lastRoundValue[-1:])
            / (np.abs(lastRoundValue[-1:]) + 1e-12),
            -1,
        )
        return np.max(errors)

    def run(self):
        if self.calculateError() < self.errorThreshold:
            return
        self.shapley()

    def shapley(self):
        pass


class TMC(Shapley):

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        baseModel: Module,
        lossFunction: Module,
        metric: Metric,
        errorThreshold: float,
        truncatedRounds: int,
    ) -> None:
        super().__init__(
            x_train,
            y_train,
            x_test,
            y_test,
            baseModel,
            lossFunction,
            metric,
            errorThreshold,
            truncatedRounds,
        )

    def shapley(self):
        self.bagScore()
        error = self.calculateError()
        with tqdm(
            total=self.truncatedRounds,
            desc=f"Calculating TMC shapley, error={error:.4f}",
        ) as t:
            while error > self.errorThreshold:
                for _ in range(self.truncatedRounds):
                    marginals, indexes = self.oneRound()
                    self.memory.append(marginals)
                    self.indexes.append(indexes)
                    t.update()

                error = self.calculateError()
                t.refresh()
                t.reset()
                t.set_description(f"Calculating TMC shapley, error={error:.4f}")

    def oneRound(self):
        metric = self.metric
        model = deepcopy(self.baseModel)
        lossFunction = self.lossFunction
        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test
        indexes = np.random.permutation(len(x_train))
        marginalContributions = np.zeros(len(x_train))
        x = torch.zeros((0,) + tuple(x_train.shape[1:])).to(x_train.device)
        y = torch.zeros((0,) + tuple(y_train.shape[1:])).to(y_train.device)
        truncationCount = 0
        newScore = metric(model(x_test), y_test)
        setSize = len(torch.unique(y_train))
        with tqdm(indexes, desc="TMC one round", leave=False) as t:
            for i in t:
                oldScore = newScore
                x = torch.cat([x, x_train[i].unsqueeze(0)])
                y = torch.cat([y, y_train[i].unsqueeze(0)])
                if len(torch.unique(y)) == setSize:
                    model = deepcopy(self.baseModel)

                    model = trainModel(model, x, y, lossFunction)
                    newScore = metric(model(x_test), y_test)
                marginalContributions[i] = (newScore - oldScore).detach()
                distanceToFullScore = torch.abs(newScore - self.mean)
                if distanceToFullScore < self.errorThreshold * self.mean:
                    truncationCount += 1
                    if truncationCount >= 5:
                        break
                else:
                    truncationCount = 0
                t.set_postfix(truncationCount=truncationCount)
        return marginalContributions, indexes
