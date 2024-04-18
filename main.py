import numpy as np
import torch
from torchmetrics.classification import Accuracy
from copy import deepcopy
from tqdm import trange
from data_utils import f, regression2classification
from model import LogisticRegressionModel, trainModel
from Shapley import looScore, TMC, G

if __name__ == "__main__":

    # settings
    POLY_DEGREE = 2  # the order of the polynomial
    X_DIMENSION = 50  # the dimension of x
    IMPORTANT = 2
    SIZE = 100  # the size of train dataset

    torch.set_num_threads(6)
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
                np.zeros(X_DIMENSION), np.eye(X_DIMENSION), size=SIZE + 5000
            )
            x_true = torch.tensor(x_true, dtype=torch.float32) * 5
            w_dimension = sum([IMPORTANT**pd for pd in range(1, POLY_DEGREE + 1)])
            w_true = torch.rand(size=[1, w_dimension])
            b_true = torch.rand(1) * 5
            y_true = f(x_true, w_true, b_true, POLY_DEGREE, IMPORTANT)
            y_true = regression2classification(y_true, parameter=parameter)
            # print("w_true:", w_true.shape)
            # print("x_true:", x_true.shape)
            # print("b_true:", b_true.shape)
            # print("y_true:", y_true.shape)

            # split dataset
            x_train = x_true[:SIZE].to(device)
            y_train = y_true[:SIZE].to(device)
            x_test = x_true[SIZE:].to(device)
            y_test = y_true[SIZE:].to(device)

            # train
            model = trainModel(
                model=deepcopy(baseModel),
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

    # print("w_pred:", model.linear.weight.view(-1))
    # print("w_true:", w_true)
    # print("b_pred:", model.linear.bias.view(-1))
    # print("b_true:", b_true)
    # print("R2Score:", baseScore)

    # Leave-One-Out
    LOOScore = looScore(
        x_train,
        y_train,
        x_test,
        y_test,
        baseModel,
        lossFunction,
        metric,
        baseScore,
    )

    errorThreshold = 0.1
    # TMC
    # average performance and error

    tmc = TMC(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        baseModel=baseModel,
        lossFunction=lossFunction,
        metric=metric,
        errorThreshold=errorThreshold,
        truncatedRounds=100,
        seed=0,
    )
    tmc.shapley()
    tmc.plotFigure([tmc.values, LOOScore])

    g = G(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        baseModel=baseModel,
        lossFunction=lossFunction,
        metric=metric,
        errorThreshold=errorThreshold,
        truncatedRounds=100,
        epoch=1,
        seed=0,
    )
    g.shapley()
    g.plotFigure([g.values, LOOScore])
