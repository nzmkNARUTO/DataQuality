import time
import numpy as np
import torch
import torch.multiprocessing as mp
from torchmetrics.classification import Accuracy
from copy import deepcopy
from tqdm import trange
from torchviz import make_dot
from data_utils import f, regression2classification, plot_figure
from model import LogisticRegressionModel, train_model
from shapley import TMCShapley, GShapley, DShapley, LOO

import sys

sys.dont_write_bytecode = True

if __name__ == "__main__":

    # settings
    POLY_DEGREE = 2  # the order of the polynomial
    X_DIMENSION = 50  # the dimension of x
    IMPORTANT = 2
    TRAIN_SIZE = 100  # the size of train dataset
    TEST_SIZE = 1000  # the size of test dataset
    D_SIZE = 1000
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
                size=TRAIN_SIZE + TEST_SIZE + D_SIZE,
            )
            x_true = torch.tensor(x_true, dtype=torch.float32) * 5
            w_dimension = sum([IMPORTANT**pd for pd in range(1, POLY_DEGREE + 1)])
            w_true = torch.rand(size=[1, w_dimension])
            b_true = torch.rand(1) * 5
            y_true = f(x_true, w_true, b_true, POLY_DEGREE, IMPORTANT)
            y_true = regression2classification(y_true, parameter=parameter)

            # split dataset
            x_train = x_true[:TRAIN_SIZE].to(device)
            y_train = y_true[:TRAIN_SIZE].to(device)
            x_distribution = x_true[TRAIN_SIZE : TRAIN_SIZE + D_SIZE].to(device)
            y_distribution = y_true[TRAIN_SIZE : TRAIN_SIZE + D_SIZE].to(device)
            x_test = x_true[TRAIN_SIZE + D_SIZE :].to(device)
            y_test = y_true[TRAIN_SIZE + D_SIZE :].to(device)
            # train
            model = train_model(
                baseModel=deepcopy(baseModel),
                x=x_train,
                y=y_train,
                loss_func=lossFunction,
            )
            # evaluate
            y_pred = model(x_test)
            baseScore = metric(y_pred, y_test)
            t.set_postfix(score=baseScore.item(), parameter=parameter)
            if baseScore > 0.7:
                break
            parameter *= 1.01
        make_dot(y_pred).render(filename="model", view=False)

    loo = LOO(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        baseModel=baseModel,
        lossFunction=lossFunction,
        metric=metric,
    )
    loo.loo()
    loo.plot()

    errorThreshold = 0.1

    # TMC
    tmc = TMCShapley(
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
    startTime = time.time()
    tmc.shapley()
    print("TMC time:", time.time() - startTime)

    # Gradient shapley
    g = GShapley(
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
    startTime = time.time()
    g.shapley()
    print("Gradient time:", time.time() - startTime)

    d = DShapley(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_dist=x_distribution,
        y_dist=y_distribution,
        baseModel=baseModel,
        lossFunction=lossFunction,
        metric=metric,
        errorThreshold=errorThreshold,
        truncatedRounds=100,
        seed=0,
        truncatedNumber=200,
    )
    startTime = time.time()
    d.shapley()
    print("Distribution time:", time.time() - startTime)

    plot_figure(
        values={
            "TMC": tmc.values,
            "GShapley": g.values,
            "DShapley": d.values,
            "LOO": loo.values,
        },
        xTrain=x_train,
        yTrain=y_train,
        xTest=x_test,
        yTest=y_test,
        baseModel=baseModel,
        loss_func=lossFunction,
        metric=metric,
    )
