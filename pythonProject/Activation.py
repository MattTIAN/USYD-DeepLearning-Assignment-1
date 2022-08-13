from abc import ABC, abstractmethod

import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.forward(z)

    @abstractmethod
    def forward(self, z):
        """
        Forward propagation, obtain a through the activation function
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        """
        Get gradient from back
        """
        raise NotImplementedError


class ReLU(ActivationBase):
    """
    ReLU function unit
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ReLU"

    def forward(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x, **kwargs):
        return (x > 0).astype(int)




