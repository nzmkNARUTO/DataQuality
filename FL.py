import numpy as np
import torch
import torch.multiprocessing as mp
from torchmetrics.classification import Accuracy
from copy import deepcopy
from tqdm import trange
from data_utils import f, regression2classification, plotFigure
from model import LogisticRegressionModel, trainModel
from shapley import looScore, TMCShapley, GShapley, DShapley
from node import Client, Server

import sys

sys.dont_write_bytecode = True

if __name__ == "__main__":

    # settings
    POLY_DEGREE = 2  # the order of the polynomial
    X_DIMENSION = 50  # the dimension of x
    IMPORTANT = 2
    TRAIN_SIZE = 200  # the size of train dataset
    TEST_SIZE = 1000  # the size of test dataset
    D_SIZE = 1000
    CLIENT_NUMBER = 2
    torch.set_num_threads(6)
    mp.set_start_method("forkserver", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    baseModel = LogisticRegressionModel(X_DIMENSION).to(device)

    metric = Accuracy("binary").to(device)
    lossFunction = torch.nn.BCELoss().to(device)

    # train
    with trange(100, desc="Finding appropriate dataset") as t:
        parameter = 1.0
        for _ in t:
            x_true = np.random.multivariate_normal(
                np.zeros(X_DIMENSION),
                np.eye(X_DIMENSION),
                size=(TRAIN_SIZE + TEST_SIZE) * CLIENT_NUMBER + D_SIZE,
            )
            x_true = torch.tensor(x_true, dtype=torch.float32) * 5
            w_dimension = sum([IMPORTANT**pd for pd in range(1, POLY_DEGREE + 1)])
            w_true = torch.rand(size=[1, w_dimension])
            b_true = torch.rand(1) * 5
            y_true = f(x_true, w_true, b_true, POLY_DEGREE, IMPORTANT)
            y_true = regression2classification(y_true, parameter=parameter)

            # split dataset
            x_train = x_true[: TRAIN_SIZE * CLIENT_NUMBER].to(device)
            y_train = y_true[: TRAIN_SIZE * CLIENT_NUMBER].to(device)
            x_test = x_true[TRAIN_SIZE * CLIENT_NUMBER : -D_SIZE].to(device)
            y_test = y_true[TRAIN_SIZE * CLIENT_NUMBER : -D_SIZE].to(device)
            x_distribution = x_true[-D_SIZE:].to(device)
            y_distribution = y_true[-D_SIZE:].to(device)
            # train
            model = trainModel(
                baseModel=deepcopy(baseModel),
                x=x_train,
                y=y_train,
                lossFunction=lossFunction,
            )
            # evaluate
            y_pred = model(x_test)
            baseScore = metric(y_pred, y_test)
            t.set_postfix(score=baseScore.item(), parameter=parameter)
            if baseScore > 0.7:
                break
            parameter *= 1.01

    server = Server(
        baseModel=baseModel,
        x_test=x_test,
        y_test=y_test,
        x_distribution=x_distribution,
        y_distribution=y_distribution,
        metric=metric,
        threshold=0.8,
    )
    for i in range(CLIENT_NUMBER):
        x_train_temp = x_train[i * TRAIN_SIZE : (i + 1) * TRAIN_SIZE].to(device)
        y_train_temp = y_train[i * TRAIN_SIZE : (i + 1) * TRAIN_SIZE].to(device)
        x_test_temp = x_test[i * TEST_SIZE : (i + 1) * TEST_SIZE].to(device)
        y_test_temp = y_test[i * TEST_SIZE : (i + 1) * TEST_SIZE].to(device)
        client = Client(
            baseModel=deepcopy(baseModel),
            x_train=x_train_temp,
            y_train=y_train_temp,
            x_test=x_test_temp,
            y_test=y_test_temp,
            lossFunction=lossFunction,
            metric=metric,
            errorThreshold=0.1,
            truncatedRounds=100,
            seed=0,
            truncatedNumber=200,
        )
        server.clients.append(client)

    server.train()
