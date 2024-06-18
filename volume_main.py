import sys
import torch
import torch.multiprocessing as mp
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanSquaredError
from copy import deepcopy
from data_utils import (
    get_synthetic_datasets,
    generate_linear_labels,
    get_mnist_datasets,
)
from volume import Volume
from model import LogisticRegressionModel, trainModel

sys.dont_write_bytecode = True

if __name__ == "__main__":
    n_participants = M = 3
    TRAINSIZE = [200] * M
    TESTSIZE = [200] * M
    D = 6

    x_trains = get_synthetic_datasets(n_participants=M, d=D, sizes=TRAINSIZE)
    x_tests = get_synthetic_datasets(n_participants=M, d=D, sizes=TESTSIZE)
    y_trains, w_trues, b_trues = generate_linear_labels(x_trains, d=D)
    y_tests, _, _ = generate_linear_labels(x_tests, d=D, weights=w_trues, bias=b_trues)
    # x_trains, x_tests = get_mnist_datasets(n_participants=M)

    volume = Volume(x_trains=x_trains, omega=0.1)
    volume.computeVolume()
    volume.computeRobustVolumes()
    print(volume.volumes)
    print(volume.robust_volumes)

    mp.set_start_method("forkserver", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    baseModel = LogisticRegressionModel(D).to(device)
    metric = MeanSquaredError().to(device)
    lossFunction = torch.nn.MSELoss().to(device)

    mses = []
    for i in range(M):
        x_train = x_trains[i].to(device)
        y_train = y_trains[i].to(device)
        x_test = x_tests[i].to(device)
        y_test = y_tests[i].to(device)
        model = trainModel(
            baseModel=deepcopy(baseModel),
            x=x_train,
            y=y_train,
            lossFunction=lossFunction,
        )
        y_pred = model(x_test)
        mse = metric(y_pred, y_test)
        mses.append(mse)
    print(mses)
