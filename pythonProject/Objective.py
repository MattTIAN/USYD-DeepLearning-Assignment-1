from abc import ABC, abstractmethod

import numpy as np


class ObjectiveBase(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def loss(self, y_true, y_pred):
        """
        Calculate loss
        """
        raise NotImplementedError

    @abstractmethod
    def grad(self, y_true, y_pred, **kwargs):
        """
        Calculate the gradient of the cost function
        """
        raise NotImplementedError


class SquaredError(ObjectiveBase):
    """
    SEC function
    """

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self):
        return "SquaredError"

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true：The true values of the n samples trained in the shape of a (n,m) array.
        :param y_pred：The predicted values of the n samples trained in the shape of a (n,m) array.
        """
        (n, _) = y_true.shape
        return 0.5 * np.linalg.norm(y_pred - y_true) ** 2 / n

    @staticmethod
    def grad(y_true, y_pred, z, acti_fn):
        (n, _) = y_true.shape
        return (y_pred - y_true) * acti_fn.grad(z) / n


class CrossEntropy(ObjectiveBase):
    """
    Cross-entropy cost function
    """

    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)

    def __str__(self):
        return "CrossEntropy"

    @staticmethod
    def loss(y_true, y_pred):
        """
        :param y_true：The true values of the n samples trained, which are required to be shaped as (n,m) binary (each sample is one-hot encoded).
        :param y_pred：The predicted values of the n samples trained in the shape of (n,m).
        """
        (n, _) = y_true.shape
        eps = np.finfo(float).eps  # Prevent np.log(0)
        cross_entropy = -np.sum(y_true * np.log(y_pred + eps)) / n
        return cross_entropy

    @staticmethod
    def grad(y_true, y_pred):
        (n, _) = y_true.shape
        grad = (y_pred - y_true) / n
        return grad
