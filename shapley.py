import math
from time import sleep
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from torch.nn.modules import Module
from torchmetrics.metric import Metric
from copy import deepcopy
from tqdm import tqdm, trange
from model import train_model

import sys

sys.dont_write_bytecode = True

MULTIPROCESS = True
MAXCPUCOUNT = 20


class LOO:

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        baseModel: torch.nn.Module,
        lossFunction: torch.nn.Module,
        metric: Metric,
    ) -> None:
        self.x_train = x_train  # the input data of training dataset
        self.y_train = y_train  # the target data of training dataset
        self.x_test = x_test  # the input data of testing dataset
        self.y_test = y_test  # the target data of testing dataset
        self.baseModel = baseModel  # the base model
        self.lossFunction = lossFunction  # the loss function
        self.metric = metric  # the metric function
        self.values = []
        y_pred = self.baseModel(x_test)
        self.baseScore = metric(y_pred, y_test)

    def _looOneRound(self, i):
        model = train_model(
            baseModel=deepcopy(self.baseModel),
            x=torch.cat(
                [self.x_train[:i], self.x_train[i + 1 :]]
            ),  # train model without i-th data
            y=torch.cat([self.y_train[:i], self.y_train[i + 1 :]]),
            loss_func=self.lossFunction,
            tqdm=False,
        )  # train model without i-th data
        y_pred = model(self.x_test)
        score = self.metric(y_pred, self.y_test)
        return score, model.state_dict()

    def loo(self):
        cpuNumber = min(mp.cpu_count() - 1, MAXCPUCOUNT)
        with tqdm(range(len(self.x_train)), desc="Caculating LOO") as t:
            if MULTIPROCESS:
                processes = []
                pool = mp.Pool(cpuNumber)
                update = lambda *args: t.update()
                for i in range(len(self.x_train)):
                    processes.append(
                        pool.apply_async(
                            func=self._looOneRound, args=(i,), callback=update
                        )
                    )
                pool.close()
                pool.join()
            else:
                self.values = [self._looOneRound(i) for i in t]
            modelsParams = []
            for p in processes:
                score, modelParams = p.get()
                self.values.append((self.baseScore - score).cpu().detach().numpy())
                modelsParams.append(modelParams)
            keys = modelsParams[0].keys()
            newModelParams = {}
            for key in keys:
                newModelParams[key] = torch.stack(
                    [modelParams[key] for modelParams in modelsParams]
                ).mean(0)
            self.modelsParams = newModelParams

    def plot(self):
        plt.plot(
            range(len(self.values)),
            self.values,
            "-",
            lw=5,
            ms=10,
        )
        plt.savefig(f"test.png", bbox_inches="tight")
        plt.close()


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
        self.modelParams = None  # the parameters of the models
        if MULTIPROCESS:  # whether to use multiprocessing
            self.tqdm = False
        else:
            self.tqdm = True

    def _oneRound(self):
        raise NotImplementedError

    def _shapley(self):
        """
        Calculate the Shapley values
        """
        className = self.__class__.__name__
        round = 0
        error = self._calculateError()
        cpuNumber = min(mp.cpu_count() - 1, MAXCPUCOUNT)
        modelsParams = []
        with tqdm(
            total=self.truncatedRounds,
            desc=f"Calculating {className} shapley round {round}, error={error:.4f}",
            leave=True,
        ) as t:
            update = lambda *args: t.update()
            while error > self.errorThreshold:
                t.reset()  # reset the progress bar for reusing
                if MULTIPROCESS:
                    processes = []
                    pool = mp.Pool(cpuNumber)
                    for _ in range(self.truncatedRounds):
                        processes.append(
                            pool.apply_async(func=self._oneRound, callback=update)
                        )

                    pool.close()
                    pool.join()
                    for p in processes:
                        marginals, indexes, modelParams = p.get()
                        self.memory.append(marginals)
                        self.indexes.append(indexes)
                else:
                    for _ in range(self.truncatedRounds):
                        marginals, indexes, modelParams = self._oneRound()
                        self.memory.append(marginals)
                        self.indexes.append(indexes)
                        update()

                error = self._calculateError()
                round += 1
                t.set_description(
                    f"Calculating {className} shapley round {round}, error={error:.4f}"
                )
                t.refresh()
            modelsParams.append(modelParams)
        keys = modelsParams[0].keys()
        newModelParams = {}
        for key in keys:
            newModelParams[key] = torch.stack(
                [modelParams[key] for modelParams in modelsParams]
            ).mean(0)
        self.modelParams = newModelParams
        self.values = np.mean(self.memory, axis=0)

    def _bagScore(self) -> tuple[float, float]:
        """
        Calculate Bagging mean&std

        Returns:
        --------
        return: tuple[float, float]
            the Bagging [std, mean]
        """

        model = train_model(
            baseModel=deepcopy(self.baseModel),
            x=self.x_train,
            y=self.y_train,
            loss_func=self.lossFunction,
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


class TMCShapley(Shapley):

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
                model = train_model(
                    baseModel=deepcopy(self.baseModel),
                    x=x,
                    y=y,
                    loss_func=self.lossFunction,
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
        return marginalContributions, indexes, model.state_dict()


class GShapley(Shapley):
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
            model = train_model(
                baseModel=deepcopy(self.baseModel),
                x=self.x_train,
                y=self.y_train,
                loss_func=self.lossFunction,
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
                processes = []
                pool = mp.Pool(cpuNumber)
                update = lambda *args: t.update()
                for i in np.arange(1, 5, 0.5):
                    processes.append(
                        pool.apply_async(
                            func=self._testAllLearningRate, args=(i,), callback=update
                        )
                    )
                pool.close()
                pool.join()
                for p in processes:
                    result, i = p.get()
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
                        model = train_model(
                            baseModel=deepcopy(self.baseModel),
                            x=self.x_train,
                            y=self.y_train,
                            loss_func=self.lossFunction,
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
                model = train_model(
                    baseModel=model,
                    x=self.x_train[batch],
                    y=self.y_train[batch],
                    loss_func=self.lossFunction,
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
        return individualContributions, indexes, model.state_dict()

    def shapley(self):
        """

        Calculate the G Shapley values

        """
        self.learningRate = self._findLearningRate()
        print(f"Using learning rate {self.learningRate}")

        self._shapley()


class DShapley(Shapley):

    def __init__(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x_dist: torch.Tensor,
        y_dist: torch.Tensor,
        baseModel: Module,
        lossFunction: Module,
        metric: Metric,
        errorThreshold: float,
        truncatedRounds: int,
        truncatedNumber: int,
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
        self.x_dist = x_dist
        self.y_dist = y_dist
        self.truncatedNumber = truncatedNumber

    def _oneRound(self):
        model = deepcopy(self.baseModel)
        marginalContributions = np.zeros(len(self.x_train))
        k = np.random.choice(np.arange(1, self.truncatedNumber + 1))
        if k == 1:
            return marginalContributions, [], model.state_dict()
        s = np.random.choice(len(self.x_dist), k - 1)
        x, y = self.x_dist[s], self.y_dist[s]
        model = train_model(
            baseModel=model,
            x=x,
            y=y,
            loss_func=self.lossFunction,
            tqdm=self.tqdm,
        )
        y_pred = model(self.x_test)
        initScore = self.metric(y_pred, self.y_test)
        if self.tqdm:
            t = tqdm(range(len(self.x_train)), desc="DShapley one round", leave=False)
        else:
            t = range(len(self.x_train))
        for i in t:
            x = torch.cat([x, self.x_train[i].unsqueeze(0)])
            y = torch.cat([y, self.y_train[i].unsqueeze(0)])
            model = deepcopy(self.baseModel)
            model = train_model(
                baseModel=model,
                x=x,
                y=y,
                loss_func=self.lossFunction,
                tqdm=self.tqdm,
            )
            score = self.metric(model(self.x_test), self.y_test)
            marginalContributions[i] = (score - initScore).detach().cpu().numpy()
            if self.tqdm:
                t.set_postfix(score=marginalContributions[i])
        return marginalContributions, list(s), model.state_dict()

    def shapley(self):
        self._shapley()
