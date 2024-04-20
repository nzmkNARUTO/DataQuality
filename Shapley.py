import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.nn.modules import Module
from torchmetrics.metric import Metric
from copy import deepcopy
from tqdm import tqdm, trange
from model import trainModel

MULTIPROCESS = True
MAXCPUCOUNT = 25


def looScore(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    baseModel: torch.nn.Module,
    lossFunction: torch.nn.Module,
    metric: Metric,
    baseScore: float,
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
    baseScore: float
        the base performance score

    Returns:
    --------
    return: list[float]
        the Leave-One-Out score
    """

    LOOScore = []
    for i in trange(len(x_train), desc="Caculating LOO"):
        model = trainModel(
            baseModel=deepcopy(baseModel),
            x=torch.cat(
                [x_train[:i], x_train[i + 1 :]]
            ),  # train model without i-th data
            y=torch.cat([y_train[:i], y_train[i + 1 :]]),
            lossFunction=lossFunction,
        )  # train model without i-th data
        y_pred = model(x_test)
        Score = metric(y_pred, y_test)
        LOOScore.append((baseScore - Score).cpu().detach().numpy())
    return LOOScore


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
        seed: int = 0,
    ) -> None:
        """

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
            the metric function, accuracy
        errorThreshold: float
            the error threshold
        truncatedRounds: int
            truncate every truncatedRounds
        seed: int
            the seed

        """
        self.x_train = x_train  # the input data of training dataset
        self.y_train = y_train  # the target data of training dataset
        self.x_test = x_test  # the input data of testing dataset
        self.y_test = y_test  # the target data of testing dataset
        self.baseModel = baseModel  # the base model
        self.lossFunction = lossFunction  # the loss function
        self.metric = metric  # the metric function, accuracy
        self.errorThreshold = errorThreshold  # the error threshold
        self.truncatedRounds = truncatedRounds  # truncate every truncatedRounds
        np.random.seed(seed)  # random seed
        self.memory = []  # the memory of the marginal contributions
        self.indexes = []  # the indexes of the data
        self.std = None  # the standard deviation of the Bagging
        self.mean = None  # the mean of the Bagging
        self.values = None  # the Shapley values
        if MULTIPROCESS:  # whether to use multiprocessing
            self.tqdm = False
        else:
            self.tqdm = True

    def _oneRound(self):
        raise NotImplementedError

    def _shapley(self):
        className = self.__class__.__name__
        round = 1
        error = self._calculateError()
        processes = []
        cpuNumber = min(mp.cpu_count(), MAXCPUCOUNT)
        with tqdm(
            total=self.truncatedRounds,
            desc=f"Calculating {className} shapley round {round}, error={error:.4f}",
            leave=True,
        ) as t:
            update = lambda *args: t.update()
            while error > self.errorThreshold:
                t.reset()  # reset the progress bar for reusing
                if MULTIPROCESS:
                    pool = mp.Pool(cpuNumber)
                    for _ in range(self.truncatedRounds):
                        processes.append(
                            pool.apply_async(func=self._oneRound, callback=update)
                        )

                    pool.close()
                    pool.join()
                    for p in processes:
                        marginals, indexes = p.get()
                        self.memory.append(marginals)
                        self.indexes.append(indexes)
                else:
                    for _ in range(self.truncatedRounds):
                        marginals, indexes = self._oneRound()
                        self.memory.append(marginals)
                        self.indexes.append(indexes)
                        update()

                error = self._calculateError()
                t.set_description(
                    f"Calculating {className} shapley round {round}, error={error:.4f}"
                )
                t.refresh()
                round += 1
        self.values = np.mean(self.memory, axis=0)

    def _calculatePortionPerformance(self, indexes: list, plotPoints: list) -> np.array:
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
            torch.max(
                torch.bincount(self.y_test.squeeze(-1).int()).float() / len(self.y_test)
            )
            .detach()
            .cpu()
        )
        for i in trange(
            len(plotPoints), 0, -1, desc="Calculating portion performance", leave=False
        ):
            keepIndexes = np.array(
                [idx for idx in indexes[plotPoints[i - 1] :]]
            )  # every plotPoints data
            x, y = self.x_train[keepIndexes], self.y_train[keepIndexes]
            if len(torch.unique(y)) == len(
                torch.unique(self.y_test)
            ):  # if select data contains all classes
                model = trainModel(
                    baseModel=deepcopy(self.baseModel),
                    x=x,
                    y=y,
                    lossFunction=self.lossFunction,
                    tqdm=self.tqdm,
                )
                y_pred = model(self.x_test)
                scores.append(self.metric(y_pred, self.y_test).detach().cpu())
            else:
                scores.append(initScore)
        return np.array(scores)[::-1]

    def plotFigure(self, values: list, numsOfPlotMarkers: int = 20) -> None:
        """
        Plot the result figure

        Parameters:
        -----------
        values: list
            the values need to be plotted
        numsOfPlotMarkers: int
            the number of plot markers

        """
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
            self._calculatePortionPerformance(np.argsort(valueSource)[::-1], pointsPlot)
            for valueSource in tqdm(valueSources, desc="Calculating performance")
        ]
        random = np.mean(
            [
                self._calculatePortionPerformance(
                    np.random.permutation(
                        np.argsort(valueSources[0])[::-1]
                    ),  # random exclude 1 datum
                    pointsPlot,
                )
                for _ in trange(10, desc="Calculating random performance")
            ],
            0,
        )
        plt.plot(
            pointsPlot / len(self.x_train) * 100,
            performance[0] * 100,
            "-",
            lw=5,
            ms=10,
            color="b",
            label=f"{self.__class__.__name__}",
        )
        plt.plot(
            pointsPlot / len(self.x_train) * 100,
            performance[-1] * 100,
            "-.",
            lw=5,
            ms=10,
            color="g",
            label="LOO",
        )
        plt.plot(
            pointsPlot / len(self.x_train) * 100,
            random * 100,
            ":",
            lw=5,
            ms=10,
            color="r",
            label="Random",
        )
        plt.legend()
        plt.savefig(f"{self.__class__.__name__}.png", bbox_inches="tight")
        plt.close()

    def _bagScore(self) -> tuple[float, float]:
        """
        Calculate Bagging mean&std

        Returns:
        --------
        return: tuple[float, float]
            the Bagging [std, mean]
        """

        model = trainModel(
            baseModel=deepcopy(self.baseModel),
            x=self.x_train,
            y=self.y_train,
            lossFunction=self.lossFunction,
            tqdm=self.tqdm,
        )
        bagScore = []
        for _ in trange(100, desc="Calculating Bagging std&mean"):
            bagIndexes = np.random.choice(
                len(self.y_test), len(self.y_test)
            )  # random extraction with replacement
            y_pred = model(self.x_test[bagIndexes])
            score = self.metric(y_pred, self.y_test[bagIndexes])
            bagScore.append(score)
        self.std, self.mean = torch.std_mean(torch.stack(bagScore))

    def _calculateError(self) -> float:
        """
        Calculate the error

        Returns:
        --------
        return: float
            the error
        """

        if len(self.memory) < self.truncatedRounds:  # not finish calculating
            return 1.0
        meanEveryRound = np.cumsum(self.memory, axis=0) / np.reshape(
            np.arange(1, len(self.memory) + 1), (-1, 1)
        )  # historical average
        errors = np.mean(
            np.abs(meanEveryRound[-self.truncatedRounds :] - meanEveryRound[-1:])
            / (np.abs(meanEveryRound[-1:]) + 1e-12),
            -1,
        )  # calculate the ratio of the difference between the historical average and the current average
        return np.max(errors)


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
        seed: int = 0,
    ) -> None:
        """

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
            the metric function, accuracy
        errorThreshold: float
            the error threshold
        truncatedRounds: int
            truncate every truncatedRounds
        seed: int
            the seed

        """
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            baseModel=baseModel,
            lossFunction=lossFunction,
            metric=metric,
            errorThreshold=errorThreshold,
            truncatedRounds=truncatedRounds,
            seed=seed,
        )

    def shapley(self):
        """
        Calculate the TMC Shapley values
        """
        self._bagScore()
        self._shapley()

    def _oneRound(self):
        """

        Calculate the marginal contributions

        Returns:
        --------
        return: tuple[np.array, list]
            the marginal contributions and the indexes

        """
        model = deepcopy(self.baseModel)
        indexes = np.random.permutation(
            len(self.x_train)
        )  # random permutation of train data by choicing random permuted indexes
        marginalContributions = np.zeros(len(self.x_train))
        x = torch.zeros((0,) + tuple(self.x_train.shape[1:])).to(self.x_train.device)
        y = torch.zeros((0,) + tuple(self.y_train.shape[1:])).to(self.y_train.device)
        truncationCount = 0
        newScore = self.metric(self.baseModel(self.x_test), self.y_test)
        setSize = len(torch.unique(self.y_train))
        if self.tqdm:
            t = tqdm(indexes, desc="TMC one round", leave=False)
        else:
            t = indexes
        for i in t:
            oldScore = newScore
            x = torch.cat([x, self.x_train[i].unsqueeze(0)])
            y = torch.cat([y, self.y_train[i].unsqueeze(0)])
            if len(torch.unique(y)) == setSize:  # if select data contains all classes
                model = trainModel(
                    baseModel=deepcopy(self.baseModel),
                    x=x,
                    y=y,
                    lossFunction=self.lossFunction,
                    tqdm=self.tqdm,
                )
            newScore = self.metric(model(self.x_test), self.y_test)
            marginalContributions[i] = (newScore - oldScore).detach()
            distanceToFullScore = torch.abs(newScore - self.mean)
            if (
                distanceToFullScore <= 0.01 * self.mean
            ):  # if the distance is less than 1% of the mean, then the rest of data is not important
                truncationCount += 1
                if truncationCount >= 4:
                    break
            else:
                truncationCount = 0
            if self.tqdm:
                t.set_postfix(
                    truncationCount=truncationCount,
                    distance=f"{distanceToFullScore.detach().cpu().numpy():.4f}",
                )
        return marginalContributions, indexes


class G(Shapley):
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
        epoch: int,
        batchSize: int = 1,
        seed: int = 0,
    ) -> None:
        """

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
            the metric function, accuracy
        errorThreshold: float
            the error threshold
        truncatedRounds: int
            truncate every truncatedRounds
        epoch: int
            the number of epoch
        batchSize: int
            the batch size
        seed: int
            the seed

        """
        super().__init__(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            baseModel=baseModel,
            lossFunction=lossFunction,
            metric=metric,
            errorThreshold=errorThreshold,
            truncatedRounds=truncatedRounds,
            seed=seed,
        )
        self.learningRate = None
        self.maxEpoch = epoch
        self.batchSize = batchSize

    def _testAllLearningRate(self, learningRate):
        """

        Parameters:
        -----------
        learningRate: float
            the learning rate

        Returns:
        --------
        return: tuple[list, float]
            the scores and the learning rate

        """
        scores = []
        for _ in range(10):
            model = trainModel(
                baseModel=deepcopy(self.baseModel),
                x=self.x_train,
                y=self.y_train,
                lossFunction=self.lossFunction,
                learningRate=10 ** (-learningRate),
                tqdm=False,
            )
            y_pred = model(self.x_test)
            score = self.metric(y_pred, self.y_test)
            scores.append(score)
        return scores, learningRate

    def _findLearningRate(self):
        """

        Returns:
        --------
        return: float
            the learning rate

        """
        bestScore = 0.0
        cpuNumber = min(mp.cpu_count(), MAXCPUCOUNT)
        with tqdm(np.arange(1, 5, 0.5), desc="Finding learning rate") as t:
            if MULTIPROCESS:
                pool = mp.Pool(cpuNumber)
                update = lambda *args: t.update()
                results = pool.map_async(
                    func=self._testAllLearningRate,
                    iterable=np.arange(1, 5, 0.5),
                    callback=update,
                )
                pool.close()
                pool.join()
                for result, i in results.get():
                    if (
                        torch.mean(torch.stack(result)) - torch.std(torch.stack(result))
                        > bestScore
                    ):
                        bestScore = torch.mean(torch.stack(result)) - torch.std(
                            torch.stack(result)
                        )
                        learningRate = 10 ** (-i)
            else:
                for i in t:
                    t.set_description(f"Finding learning rate 10e{(-i)}")
                    scores = []
                    for _ in trange(10, desc="Testing learning rate", leave=False):
                        model = trainModel(
                            baseModel=deepcopy(self.baseModel),
                            x=self.x_train,
                            y=self.y_train,
                            lossFunction=self.lossFunction,
                            learningRate=10 ** (-i),
                        )
                        y_pred = model(self.x_test)
                        score = self.metric(y_pred, self.y_test)
                        scores.append(score)
                    if (
                        torch.mean(torch.stack(scores)) - torch.std(torch.stack(scores))
                        > bestScore
                    ):
                        bestScore = torch.mean(torch.stack(scores)) - torch.std(
                            torch.stack(scores)
                        )
                        learningRate = 10 ** (-i)
        return learningRate

    def _oneRound(self):
        """

        Returns:
        --------
        return: tuple[np.array, list]
            the marginal contributions and the indexes

        """
        model = deepcopy(self.baseModel)
        marginalContributions = np.zeros(len(self.x_train))
        indexes = []
        values = []
        stopCounter = 0
        bestScore = -math.inf
        for _ in range(self.maxEpoch):
            vals = []
            idxs = np.random.permutation(len(self.x_train))
            batches = [
                idxs[k * self.batchSize : (k + 1) * self.batchSize]
                for k in range(int(np.ceil(len(idxs) / self.batchSize)))
            ]
            idxs = batches
            for _, batch in enumerate(batches):
                model = trainModel(
                    baseModel=model,
                    x=self.x_train[batch],
                    y=self.y_train[batch],
                    lossFunction=self.lossFunction,
                    learningRate=self.learningRate,
                    tqdm=self.tqdm,
                )
                vals.append(
                    self.metric(model(self.x_test), self.y_test).detach().cpu().numpy()
                )
            indexes.append(idxs)
            values.append(vals)
            currentScore = np.mean(vals)
            if currentScore > bestScore:
                bestScore = currentScore
                stopCounter = 0
            else:
                stopCounter += 1
                if stopCounter >= 1:
                    break
        marginalContributions[1:] += values[0][1:]
        marginalContributions[1:] -= values[0][:-1]
        individualContributions = np.zeros(len(self.x_train))
        for i, index in enumerate(indexes[0]):
            individualContributions[index] += marginalContributions[i]
            individualContributions[index] /= len(index)
        return individualContributions, indexes

    def shapley(self):
        """

        Calculate the G Shapley values

        """
        self.learningRate = self._findLearningRate()
        print(f"Using learning rate {self.learningRate}")

        self._shapley()
