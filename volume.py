import torch
import numpy as np
from math import ceil, floor
from collections import defaultdict, Counter
from torch import stack, cat, zeros_like, pinverse
from tqdm import tqdm
from data_utils import VolumeDataset


class Volume:
    def __init__(self, dataset, omega) -> None:
        self.x = dataset.x
        self.y = dataset.y
        self.omega = omega
        self.volume = None
        self.xTilde = None
        self.cubes = None
        self.robustVolume = None

    def compute_volume(self):
        d = self.x.shape[1]
        self.x = self.x.reshape(-1, d)

        volume = np.sqrt(np.linalg.det(self.x.T @ self.x) + 1e-8)

        self.volume = volume
        return self

    def _compute_x_tilde(self):
        """
        Compresses the original feature matrix X to  X_tilde with the specified omega.

        Returns:
        X_tilde: compressed np.ndarray
        cubes: a dictionary of cubes with the respective counts in each dcube
        """
        D = self.x.shape[1]

        # assert 0 < omega <= 1, "omega must be within range [0,1]."

        m = ceil(1.0 / self.omega)  # number of intervals for each dimension

        cubes = Counter()  # a dictionary to store the freqs
        # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
        # value: counts

        Omega = defaultdict(list)
        # Omega = {}

        minDimensions = torch.min(self.x, axis=0).values

        # a dictionary to store cubes of not full size
        for x in tqdm(self.x):
            cube = []
            for _, xd in enumerate(x - minDimensions):
                d_index = floor(xd / self.omega)
                cube.append(d_index)

            cubeKey = tuple(cube)
            cubes[cubeKey] += 1

            Omega[cubeKey].append(x)

            """
            if cube_key in Omega:
                
                # Implementing mean() to compute the average of all rows which fall in the cube
                
                Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
                # Omega[cube_key].append(x)
            else:
                Omega[cube_key] = x
            """
        xTilde = stack([stack(list(value)).mean(axis=0) for _, value in Omega.items()])

        # X_tilde = stack(list(Omega.values()))

        self.xTilde = xTilde
        self.cubes = cubes

    def compute_robust_volumes(self, lens=None):
        self._compute_x_tilde()
        if lens is None:
            N = len(self.xTilde)
        else:
            N = sum(lens)
        alpha = 1.0 / (10 * N)  # it means we set beta = 10
        # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))
        volume = (
            Volume(dataset=VolumeDataset(self.xTilde, self.y), omega=self.omega)
            .compute_volume()
            .volume
        )

        # volumes, _ = self.computeVolume(self.X_tildes)
        rhoOmegaProd = 1.0
        for _, freqCount in self.cubes.items():

            # if freq_count == 1: continue # volume does not monotonically increase with omega
            # commenting this if will result in volume monotonically increasing with omega
            rho_omega = (1 - alpha ** (freqCount + 1)) / (1 - alpha)

            rhoOmegaProd *= rho_omega

        robustVolume = (volume * rhoOmegaProd).round(3)
        self.robustVolume = robustVolume
