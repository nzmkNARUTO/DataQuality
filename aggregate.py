import torch


def fedavg(diffModelsParams, Vs):
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


def fedtest1(diffModelsParams, RVs, mask):
    keys = RVs[0].keys()
    # newRVs = [RV["rhoOmegaProd"] * torch.sqrt(torch.exp(RV["logdet"])) for RV in RVs]
    newRVs = [RV["rhoOmegaProd"] * torch.exp(RV["logdet"]) ** (1 / 5) for RV in RVs]
    print(newRVs)
    sumRV = sum(newRVs)
    keys = diffModelsParams[0].keys()
    newModelParams = {}
    for key in keys:
        newModelParams[key] = (
            torch.stack(
                [
                    diffModelParams[key] * newRVs[i]
                    for i, diffModelParams in enumerate(diffModelsParams)
                ]
            ).sum(0)
            / sumRV
        )
    return newModelParams


def fedtest(diffModelsParams, Vs, masks):
    Vs = torch.stack(Vs)
    VsV = Vs / Vs.sum(0)
    VsH = Vs / Vs.sum(1).reshape(-1, 1)
    for client in VsV:
        for c in masks:
            mask = 1
    keys = diffModelsParams[0].keys()
    newModelParams = {}
    for i in range(len(diffModelsParams)):
        for key in keys:
            diffModelsParams[i][key] = sum(
                diffModelsParams[i][key] * masks[j][key] * VsV[i][j]
                for j in range(len(masks))
            )
    for key in keys:
        newModelParams[key] = sum(
            [diffModelParams[key] for diffModelParams in diffModelsParams]
        )
    return newModelParams