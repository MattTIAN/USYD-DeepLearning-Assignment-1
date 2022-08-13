from abc import ABC, abstractmethod
import numpy as np


class OptimizerBase(ABC):
    def __init__(self):
        pass

    def __call__(self, params, params_grad, params_name):
        """
        :param params：parameters to be updated, such as the weight matrix W.
        :param params_grad：The gradient of the parameter to be updated.
        :param params_name：The name of the parameter to be updated.
        """
        return self.update(params, params_grad, params_name)

    @abstractmethod
    def update(self, params, params_grad, params_name):
        """
        :param params：parameters to be updated, such as the weight matrix W.
        :param params_grad：The gradient of the parameter to be updated.
        :param params_name：The name of the parameter to be updated.
        """
        raise NotImplementedError


class SGD(OptimizerBase):
    """
    sgd Optimization methods
    """

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr
        self.cache = {}

    def __str__(self):
        return "SGD(lr={})".format(self.hyperparams["lr"])

    def update(self, params, params_grad, params_name):
        update_value = self.lr * params_grad
        return params - update_value

    @property
    def hyperparams(self):
        return {"op": "SGD", "lr": self.lr
                }


"""
SGD momentum
"""


class Momentum(OptimizerBase):
    def __init__(
            self, lr=0.001, momentum=0.0, **kwargs
    ):
        """
        :param lr：Learning rate, float (default: 0.001)
        :param momentum：The alpha when considering Momentum, which determines how fast the previous gradient contribution decays, takes a value in the range [0, 1], default 0
        """
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.cache = {}

    def __str__(self):
        return "Momentum(lr={}, momentum={})".format(self.lr, self.momentum)

    def update(self, param, param_grad, param_name):
        C = self.cache
        lr, momentum = self.lr, self.momentum
        if param_name not in C:  # save v
            C[param_name] = np.zeros_like(param_grad)
        update = momentum * C[param_name] - lr * param_grad
        self.cache[param_name] = update
        return param + update

    @property
    def hyperparams(self):
        return {"op": "Momentum", "lr": self.lr,
                "momentum": self.momentum
                }
