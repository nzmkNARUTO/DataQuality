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
from aggregate import fedavg, fedtest


class Node:
    def __init__(
        self,
        globalModel: torch.nn.Module,
        loss_func: torch.nn.Module,
        dataset: VolumeDataset = None,
        istopk: bool = False,
    ):
        self.globalModel = globalModel
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.dataset = dataset
        self.keys = self.globalModel.state_dict().keys()
        self.istopk = istopk
        self.loss_func = loss_func

    def send(self):
        return self.globalModel.state_dict()

    def receive(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def topk(self, params: dict, ratio):
        if self.istopk:
            topkParams = {}
            for key in params.keys():
                topkParams[key] = torch.topk(
                    params[key].reshape(-1),
                    math.ceil(params[key].numel() * ratio),
                    largest=False,
                )
            return topkParams
        else:
            return params

    def untopk(self, params: dict):
        if self.istopk:
            untopkParams = {}
            for key in params.keys():
                untopkParams[key] = torch.zeros_like(
                    self.globalModel.state_dict()[key]
                ).cpu()
                untopkParams[key].reshape(-1)[params[key][1]] = params[key][0]
            return untopkParams
        else:
            return params

    def mask(self, sensitivities, ratio):
        sensitivities = self.topk(sensitivities, ratio)
        mask = {}
        for key in self.keys:
            mask[key] = torch.zeros_like(self.globalModel.state_dict()[key]).cpu()
            mask[key].reshape(-1)[sensitivities[key][1]] = 1
        return mask


class Client(Node):
    def __init__(
        self,
        globalModel: torch.nn.Module,
        loss_func: torch.nn.Module,
        metric: torch.nn.Module,
        dataset: VolumeDataset = None,
        epochs: int = 5,
        istopk: bool = False,
    ):
        super().__init__(
            globalModel=globalModel,
            dataset=dataset,
            istopk=istopk,
            loss_func=loss_func,
        )
        self.metric = metric
        self.epochs = epochs
        self.volume = Volume(dataset=self.dataset, omega=0.5)
        self.volume.compute_volume()
        self.volume.compute_robust_volumes()
        self.localModel = deepcopy(self.globalModel)
        self.diffModelParams = {}

    def train(self, round):
        self.localModel = train_model(
            baseModel=deepcopy(self.localModel),
            dataset=self.dataset,
            loss_func=self.loss_func,
            device=self.device,
            epochs=self.epochs,
            batchSize=16,
            learningRate=0.1,
        )
        self.round = round

    def receive(self, modelParams: dict):
        self.globalModel.load_state_dict(modelParams)
        self.localModel.load_state_dict(modelParams)

    def send(self):
        for key in self.keys:
            self.diffModelParams[key] = (
                self.localModel.cpu().state_dict()[key]
                - self.globalModel.state_dict()[key]
            )
        return self.diffModelParams


class Server(Node):

    def __init__(
        self,
        globalModel: torch.nn.Module,
        metric: torch.nn.Module,
        threshold: float,
        loss_func: torch.nn.Module,
        dataset: VolumeDataset = None,
        aggregation_method: str = "fedavg",
        istopk: bool = False,
    ):
        super().__init__(
            globalModel=globalModel,
            dataset=dataset,
            istopk=istopk,
            loss_func=loss_func,
        )
        self.metric = metric
        self.clients = []
        self.threshold = threshold
        self.aggregation_method = aggregation_method

    def receive(self, clients):
        sensitivities = []
        for i in range(self.dataset.y.max() + 1):
            sensitivity = {}
            output = self.globalModel(self.dataset.x)
            loss = self.loss_func(
                output, torch.tensor(torch.zeros_like(self.dataset.y) + i)
            )
            self.globalModel.zero_grad()
            loss.backward()
            for name, param in self.globalModel.named_parameters():
                sensitivity.update({name: param.grad})
            sensitivities.append(sensitivity)

        Vs = [torch.bincount(client.dataset.y).to(torch.float32) for client in clients]
        diffModelsParams = [client.send() for client in clients]
        # RVs = [client.volume.robustVolume for client in clients]
        if self.aggregation_method == "fedtest":
            diffModelParams = self.fedtest(diffModelsParams, Vs, sensitivities)
        elif self.aggregation_method == "fedavg":
            diffModelParams = self.fedavg(diffModelsParams, Vs)
        newModelParams = {}
        for key in self.keys:
            newModelParams[key] = (
                self.globalModel.cpu().state_dict()[key] + diffModelParams[key]
            )
        self.globalModel.load_state_dict(newModelParams)

    def fedavg(self, diffModelsParams, Vs):
        Vs = torch.sum(torch.stack(Vs), dim=1)
        sumV = sum(Vs)
        keys = diffModelsParams[0].keys()
        newModelParams = {}
        for key in keys:
            newModelParams[key] = (
                sum(
                    [
                        diffModelParams[key] * Vs[i]
                        for i, diffModelParams in enumerate(diffModelsParams)
                    ]
                )
                / sumV
            )
        return newModelParams

    def fedtest(self, diffModelsParams, Vs, sensitivities):
        Vs = torch.stack(Vs)
        VsV = Vs / Vs.sum(0)
        VsH = Vs.sum(1) / Vs.sum()
        keys = diffModelsParams[0].keys()
        masks = []
        for i in range(len(sensitivities)):
            masks.append(self.mask(sensitivities[i], 0.5))

        for i in range(len(diffModelsParams)):
            diffModelsParams[i] = self.merge(diffModelsParams[i], masks, VsV[i])

        newModelParams = {}
        for key in keys:
            newModelParams[key] = sum(
                [
                    diffModelsParams[i][key] * VsH[i]
                    for i in range(len(diffModelsParams))
                ]
            )
        return newModelParams

    def merge(self, diffModelParams, masks, VV):
        for key in self.keys:
            diffModelParams[key] = sum(
                diffModelParams[key] * masks[i][key] * VV[i] for i in range(len(masks))
            )
        return diffModelParams

    def evaluate(self):
        return test_model(
            dataset=self.dataset,
            model=self.globalModel,
            metric=self.metric,
            device=self.device,
            batchSize=128,
        )

    def select_clients(self, clients: list):
        self.clients = clients

    def train(self):
        metric = 0
        metrics = []
        for i in range(40):
            # while metric < self.threshold:
            for client in self.clients:
                client.receive(self.send())
                client.train(i)
            self.receive(self.clients)
            metric = float(self.evaluate().detach().cpu().numpy())
            metrics.append(metric)
        print(metrics)
        return metrics
