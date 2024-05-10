import torch

from model import trainModel


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
    ):
        super().__init__(baseModel)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.lossFunction = lossFunction

    def loadServerDataset(self):
        pass

    def train(self):
        self.baseModel = trainModel(
            baseModel=self.baseModel,
            x=self.x_train,
            y=self.y_train,
            lossFunction=self.lossFunction,
        )

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

    def receive(self, models: list):
        keys = models[0].keys()
        newParams = {}
        for key in keys:
            newParams[key] = torch.stack([model[key] for model in models]).mean(0)
        self.baseModel.load_state_dict(newParams)

    def evaluate(self):
        y_pred = self.baseModel(self.x_test)
        return self.metric(y_pred, self.y_test)

    def selectClients(self, clients: list):
        self.clients = clients

    def train(self):
        acc = 0
        while acc < self.threshold:
            for client in self.clients:
                client.receive(self.send())
                client.train()
            self.receive([client.send() for client in self.clients])
            acc = self.evaluate()
            print(f"accuracy:{acc.item()}")
