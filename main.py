import numpy as np
import torch
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
from copy import deepcopy
from tqdm import tqdm, trange
from data_utils import f, regression2classification
from model import LogisticRegressionModel, trainModel
from Shapley import looScore, TMC

# settings
POLY_DEGREE = 1  # the order of the polynomial
X_DIMENSION = 1  # the dimension of x
SIZE = 100  # the size of train dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
baseModel = LogisticRegressionModel(POLY_DEGREE).to(device)

metric = R2Score().to(device)
lossFunction = torch.nn.SmoothL1Loss()
lossFunction = torch.nn.BCELoss().to(device)

# train
t = trange(100, desc="Training")
for _ in t:
    x_true = np.random.multivariate_normal(
        np.zeros(X_DIMENSION), np.eye(X_DIMENSION), size=SIZE + 5000
    )
    x_true = torch.tensor(x_true, dtype=torch.float32) * 5
    w_dimension = sum([X_DIMENSION**pd for pd in range(1, POLY_DEGREE + 1)])
    w_true = torch.rand(size=[1, w_dimension])
    b_true = torch.rand(1) * 5
    y_true = f(x_true, w_true, b_true, POLY_DEGREE)
    y_true = regression2classification(y_true)
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
    model, loss = trainModel(
        model=deepcopy(baseModel), x=x_train, y=y_train, lossFunction=lossFunction
    )
    # evaluate
    y_pred = model(x_test)
    baseR2Score = metric(y_pred, y_test)
    t.set_postfix(loss=loss.item(), R2Score=baseR2Score.item())
    # print("Loss:", loss)
    # print("baseR2Score:", baseR2Score)
    if baseR2Score > 0.7:
        break

print("Loss:", loss)
print("w_pred:", model.linear.weight.view(-1))
print("w_true:", w_true)
print("b_pred:", model.linear.bias.view(-1))
print("b_true:", b_true)
print("R2Score:", baseR2Score)

# Leave-One-Out
LOO_R2Score = looScore(
    x_train,
    y_train,
    x_test,
    y_test,
    baseModel,
    lossFunction,
    metric,
    baseR2Score,
)

plt.hist(LOO_R2Score)
plt.savefig("LOO_R2Score.png")

# TMC
# average performance and error


errorThreshold = 0.1
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
)
tmc.shapley()
