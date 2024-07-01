import math
import torch

from copy import deepcopy
from model import train_model, test_model
from data_utils import (
    VolumeDataset,
    generate_synthetic_data,
    generate_linear_label,
    generate_weight_bias,
)
from volume import Volume


class Node:
    def __init__(
        self,
        baseModel: torch.nn.Module,
        dimension: int,
        size: int,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ):
        self.baseModel = baseModel
        self.dimension = dimension
        self.size = size
        self.weight = weight
        self.bias = bias
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = None

    def send(self):
        return self.baseModel.state_dict()

    def receive(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def generate_dataset(self):
        x = generate_synthetic_data(d=self.dimension, size=self.size)
        y = generate_linear_label(
            x, d=self.dimension, weight=self.weight, bias=self.bias
        )
        self.dataset = VolumeDataset(x, y)


class Client(Node):
    def __init__(
        self,
        baseModel: torch.nn.Module,
        loss_func: torch.nn.Module,
        metric: torch.nn.Module,
        weight: torch.Tensor,
        bias: torch.Tensor,
        dimension: int,
        size: int,
    ):
        super().__init__(
            baseModel=baseModel,
            weight=weight,
            bias=bias,
            dimension=dimension,
            size=size,
        )
        self.loss_func = loss_func
        self.metric = metric
        self.generate_dataset()
        self.volume = None

    def train(self):
        self.volume = Volume(dataset=self.dataset, omega=0.1)
        self.volume.compute_volume()
        self.volume.compute_robust_volumes()
        self.baseModel = train_model(
            baseModel=deepcopy(self.baseModel),
            dataset=self.dataset,
            loss_func=self.loss_func,
            device=self.device,
            epochs=3,
            batchSize=128,
        )

    def receive(self, modelParams: dict):
        self.baseModel.load_state_dict(modelParams)


class Server(Node):
    def __init__(
        self,
        baseModel: torch.nn.Module,
        metric: torch.nn.Module,
        dimension: int,
        size: int,
        threshold: float,
    ):
        super().__init__(
            baseModel, weight=None, bias=None, dimension=dimension, size=size
        )
        self.metric = metric
        self.clients = []
        self.threshold = threshold
        self.weight, self.bias = generate_weight_bias(d=self.dimension)
        self.generate_dataset()

    def receive(self, modelsParams: list):
        keys = modelsParams[0].keys()
        newModelParams = {}
        for key in keys:
            newModelParams[key] = torch.stack(
                [modelParams[key] for modelParams in modelsParams]
            ).mean(0)
        self.baseModel.load_state_dict(newModelParams)

    def evaluate(self):
        return test_model(
            dataset=self.dataset,
            model=self.baseModel,
            metric=self.metric,
            device=self.device,
            batchSize=128,
        )

    def select_clients(self, clients: list):
        self.clients = clients

    def train(self):
        metric = math.inf
        metrics = []
        for _ in range(10):
            # while metric > self.threshold:
            for client in self.clients:
                client.receive(self.send())
                client.train()
            self.receive([client.send() for client in self.clients])
            metric = float(self.evaluate().detach().cpu().numpy())
            metrics.append(metric)
        print(metrics)
