# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Coherent Ising Machine with chaotic feedback control algorithm."""
# pylint: disable=invalid-name
import numpy as np
from scipy.sparse import csr_matrix

from .QAIA import QAIA


class CFC(QAIA):
    r"""
    Coherent Ising Machine with chaotic feedback control algorithm.

    Reference: `Coherent Ising machines with optical error correction
    circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``0.1``.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=0.1,
    ):
        """Construct CFC algorithm."""
        super().__init__(J, h, x, n_iter, batch_size)
        self.J = csr_matrix(self.J)
        self.dt = dt
        # The number of first iterations
        self.Tr = int(0.9 * self.n_iter)
        # The number of additional iterations
        self.Tp = self.n_iter - self.Tr
        self.N = self.J.shape[0]
        # pumping parameters
        self.p = np.hstack([np.linspace(-1, 1, self.Tr), np.ones(self.Tp)])
        # target amplitude
        self.alpha = 1.0
        # coupling strength
        self.xi = np.sqrt(2 * self.N / np.sum(self.J**2))
        # rate of change of error variables
        self.beta = 0.15
        self.initialize()

    def initialize(self):
        """Initialize spin values and error variables."""
        if self.x is None:
            self.x = np.random.normal(0, 0.1, size=(self.N, self.batch_size))

        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

        self.e = np.ones_like(self.x)

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Dynamical evolution."""
        for i in range(self.n_iter):
            if self.h is None:
                z = self.xi * self.e * (self.J @ self.x)
            else:
                z = self.xi * self.e * (self.J @ self.x + self.h)
            self.x = self.x + (-self.x**3 + (self.p[i] - 1) * self.x + z) * self.dt
            self.e = self.e + (-self.beta * self.e * (z**2 - self.alpha)) * self.dt

            cond = np.abs(self.x) > 1.5
            self.x = np.where(cond, 1.5 * np.sign(self.x), self.x)
            self.e = np.where(self.e < 0.01, 0.01, self.e)
