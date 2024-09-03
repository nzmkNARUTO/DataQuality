import sys
import torch
import torch.multiprocessing as mp
import numpy as np
from torchmetrics.classification import Accuracy
from torchmetrics.regression import MeanSquaredError
from copy import deepcopy
from data_utils import (
    VolumeDataset,
    generate_synthetic_data,
    generate_linear_label,
    generate_weight_bias,
    generate_synthetic_classification_data,
)
from volume import Volume
from model import LogisticRegressionModel, ClassificationModel, train_model, test_model

sys.dont_write_bytecode = True

if __name__ == "__main__":
    TRAINSIZE = 5000
    TESTSIZE = 1000
    mp.set_start_method("fork", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    baseModel = ClassificationModel(10, 10)
    metric = Accuracy("multiclass", num_classes=10)
    loss_func = torch.nn.CrossEntropyLoss()

    dataset = generate_synthetic_classification_data(
        size=TRAINSIZE + TESTSIZE, dimension=10, classes=10
    )
    indices = list(range(TRAINSIZE + TESTSIZE))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:TRAINSIZE], indices[TRAINSIZE:]
    train_dataset, test_dataset = dataset.slice(train_indices), dataset.slice(
        test_indices
    )

    model = train_model(
        baseModel,
        train_dataset,
        loss_func,
        batchSize=256,
        epochs=1,
        tqdm=True,
        learningRate=0.01,
        device=device,
    )
    modelsParams = [model.cpu().state_dict(), baseModel.state_dict()]
    keys = modelsParams[0].keys()
    diffModelParams = {}
    for key in keys:
        diffModelParams[key] = modelsParams[0][key] - modelsParams[1][key]
    newModelParams = {}
    for key in keys:
        newModelParams[key] = baseModel.state_dict()[key] + diffModelParams[key]
    testModel = deepcopy(baseModel)
    testModel.load_state_dict(newModelParams)
    for key in keys:
        print(newModelParams[key].shape)
    print(newModelParams)
    acc = test_model(model, test_dataset, metric, batchSize=32, device=device)
    print(acc)
    acc = test_model(testModel, test_dataset, metric, batchSize=32, device=device)
    print(acc)
