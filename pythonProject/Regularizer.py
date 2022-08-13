from abc import ABC, abstractmethod

import numpy as np


class RegularBase(ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def loss(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def grad(self, **kwargs):
        raise NotImplementedError


class L2Regularizer(RegularBase):
    def __init__(self, lambd=0.001):
        super().__init__()
        self.lambd = lambd

    def loss(self, params):
        loss = 0
        for key, val in params.items():
            loss += 0.5 * np.sum(np.square(val)) * self.lambd
        return loss

    def grad(self, params):
        for key, val in params.items():
            grad = self.lambd * val
        return grad
