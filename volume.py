import torch
import numpy as np
from math import ceil, floor
from collections import defaultdict, Counter
from torch import stack, cat, zeros_like, pinverse


class Volume:
    def __init__(self, x_trains, omega) -> None:
        self.x_trains = x_trains
        self.omega = omega
        self.volumes = None
        self.volume_all = None
        self.X_tildes = None
        self.cubes = None
        self.robust_volumes = None

    def computeVolume(self):
        d = self.x_trains[0].shape[1]
        for i in range(len(self.x_trains)):
            self.x_trains[i] = self.x_trains[i].reshape(-1, d)

        X = np.concatenate(self.x_trains, axis=0).reshape(-1, d)
        volumes = np.zeros(len(self.x_trains))
        for i, dataset in enumerate(self.x_trains):
            volumes[i] = np.sqrt(np.linalg.det(dataset.T @ dataset) + 1e-8)

        volume_all = np.sqrt(np.linalg.det(X.T @ X) + 1e-8).round(3)
        self.volumes = volumes
        self.volume_all = volume_all
        return self

    def _computeXTilde(self):
        """
        Compresses the original feature matrix X to  X_tilde with the specified omega.

        Returns:
        X_tilde: compressed np.ndarray
        cubes: a dictionary of cubes with the respective counts in each dcube
        """
        X_tildes = []
        cubes = []
        for x_train in self.x_trains:
            D = x_train.shape[1]

            # assert 0 < omega <= 1, "omega must be within range [0,1]."

            m = ceil(1.0 / self.omega)  # number of intervals for each dimension

            cube = Counter()  # a dictionary to store the freqs
            # key: (1,1,..)  a d-dimensional tuple, each entry between [0, m-1]
            # value: counts

            Omega = defaultdict(list)
            # Omega = {}

            min_ds = torch.min(x_train, axis=0).values

            # a dictionary to store cubes of not full size
            for x in x_train:
                c = []
                for d, xd in enumerate(x - min_ds):
                    d_index = floor(xd / self.omega)
                    c.append(d_index)

                cube_key = tuple(c)
                cube[cube_key] += 1

                Omega[cube_key].append(x)

                """
                if cube_key in Omega:
                    
                    # Implementing mean() to compute the average of all rows which fall in the cube
                    
                    Omega[cube_key] = Omega[cube_key] * (1 - 1.0 / cubes[cube_key]) + 1.0 / cubes[cube_key] * x
                    # Omega[cube_key].append(x)
                else:
                    Omega[cube_key] = x
                """
            X_tilde = stack(
                [stack(list(value)).mean(axis=0) for _, value in Omega.items()]
            )

            # X_tilde = stack(list(Omega.values()))

            X_tildes.append(X_tilde)
            cubes.append(cube)
        self.X_tildes = X_tildes
        self.cubes = cubes

    def computeRobustVolumes(self):
        self._computeXTilde()
        N = sum([len(X_tilde) for X_tilde in self.X_tildes])
        alpha = 1.0 / (10 * N)  # it means we set beta = 10
        # print("alpha is :{}, and (1 + alpha) is :{}".format(alpha, 1 + alpha))
        volumes = (
            Volume(x_trains=self.X_tildes, omega=self.omega).computeVolume().volumes
        )
        # volumes, _ = self.computeVolume(self.X_tildes)
        robust_volumes = np.zeros_like(volumes)
        for i, (volume, hypercubes) in enumerate(zip(volumes, self.cubes)):
            rho_omega_prod = 1.0
            for _, freq_count in hypercubes.items():

                # if freq_count == 1: continue # volume does not monotonically increase with omega
                # commenting this if will result in volume monotonically increasing with omega
                rho_omega = (1 - alpha ** (freq_count + 1)) / (1 - alpha)

                rho_omega_prod *= rho_omega

            robust_volumes[i] = (volume * rho_omega_prod).round(3)
        self.robust_volumes = robust_volumes
