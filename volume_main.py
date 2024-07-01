import sys
import torch
import torch.multiprocessing as mp
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
    TRAINSIZE = 50000
    TESTSIZE = 10000
    mp.set_start_method("fork", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    baseModel = ClassificationModel(10, 10)
    metric = Accuracy("multiclass", num_classes=10)
    loss_func = torch.nn.CrossEntropyLoss()

    dataset = generate_synthetic_classification_data(
        size=TRAINSIZE + TESTSIZE, dimension=10, classes=10
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [TRAINSIZE, TESTSIZE]
    )

    model = train_model(
        baseModel,
        train_dataset,
        loss_func,
        batchSize=1024,
        epochs=10,
        tqdm=True,
        learningRate=0.01,
        device=device,
    )
    acc = test_model(model, test_dataset, metric, batchSize=32, device=device)
    print(acc)
