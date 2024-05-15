import torch

from model import trainModel
from shapley import DShapley, GShapley


class Node:
    def __init__(
        self,
        baseModel: torch.nn.Module,
    ):
        self.baseModel = baseModel

    def send(self):
        return self.baseModel.state_dict()

    def receive(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


class Client(Node):
    def __init__(
        self,
        baseModel: torch.nn.Module,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        lossFunction: torch.nn.Module,
        metric: torch.nn.Module,
        errorThreshold: float,
        truncatedRounds: int,
        seed: int,
        truncatedNumber: int,
    ):
        super().__init__(baseModel)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lossFunction = lossFunction
        self.metric = metric
        self.errorThreshold = errorThreshold
        self.truncatedRounds = truncatedRounds
        self.seed = seed
        self.truncatedNumber = truncatedNumber
        self.x_distribution = None
        self.y_distribution = None

    def loadServerDataset(
        self, x_distribution: torch.Tensor, y_distribution: torch.Tensor
    ):
        self.x_distribution = x_distribution
        self.y_distribution = y_distribution

    def train(self):
        d = GShapley(
            x_train=self.x_train,
            y_train=self.y_train,
            x_test=self.x_test,
            y_test=self.y_test,
            baseModel=self.baseModel,
            lossFunction=self.lossFunction,
            metric=self.metric,
            errorThreshold=self.errorThreshold,
            truncatedRounds=100,
            epoch=1,
            seed=0,
        )
        d.shapley()
        self.baseModel.load_state_dict(d.modelsParams)

    def receive(self, modelParams: dict):
        self.baseModel.load_state_dict(modelParams)


class Server(Node):
    def __init__(
        self,
        baseModel: torch.nn.Module,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        x_distribution: torch.Tensor,
        y_distribution: torch.Tensor,
        metric: torch.nn.Module,
        threshold: float,
    ):
        super().__init__(baseModel)
        self.x_test = x_test
        self.y_test = y_test
        self.x_distribution = x_distribution
        self.y_distribution = y_distribution
        self.metric = metric
        self.threshold = threshold
        self.clients = []

    def receive(self, modelsParams: list):
        keys = modelsParams[0].keys()
        newModelParams = {}
        for key in keys:
            newModelParams[key] = torch.stack(
                [modelParams[key] for modelParams in modelsParams]
            ).mean(0)
        self.baseModel.load_state_dict(newModelParams)

    def evaluate(self):
        y_pred = self.baseModel(self.x_test)
        return self.metric(y_pred, self.y_test)

    def selectClients(self, clients: list):
        self.clients = clients

    def train(self):
        acc = 0
        accs = []
        while acc < self.threshold:
            for client in self.clients:
                client.receive(self.send())
                client.loadServerDataset(
                    x_distribution=self.x_distribution,
                    y_distribution=self.y_distribution,
                )
                client.train()
            self.receive([client.send() for client in self.clients])
            acc = float(self.evaluate().detach().cpu().numpy())
            accs.append(acc)
            print(accs)
