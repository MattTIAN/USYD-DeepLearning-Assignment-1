import re

from Activation import *
from Optimizer import *
from Weight import *
from Regularizer import *

"""
This is an initialization module of hyper-parameters including optimizer, activation function, weight initialization, regularizer
"""


class OptimizerInitializer(ABC):

    def __init__(self, opti_name="sgd"):
        self.opti_name = opti_name

    def __call__(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        opti_str = self.opti_name.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, opti_str)])
        if "sgd" in opti_str:
            optimizer = SGD(**kwargs)
            return optimizer
        elif "momentum" in opti_str:
            optimizer = Momentum(**kwargs)
            return optimizer


class ActivationInitializer(object):

    def __init__(self, acti_name='relu'):
        self.acti_name = acti_name

    def __call__(self):
        if self.acti_name.lower() == 'relu':
            acti_fn = ReLU()
        return acti_fn


class WeightInitializer(object):
    def __init__(self, mode="std_normal"):
        self.mode = mode
        r = r"([a-zA-Z]*)=([^,)]*)"
        mode_str = self.mode.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, mode_str)])
        if "std_normal" in mode_str:
            self.init_fn = std_normal(**kwargs)

    def __call__(self, weight_shape):
        W = self.init_fn(weight_shape)
        return W


class RegularizerInitializer(object):
    def __init__(self, regular_name='l2'):
        self.regular_name = regular_name

    def __call__(self):
        r = r"([a-zA-Z]*)=([^,)]*)"
        regular_str = self.regular_name.lower()
        kwargs = dict([(i, eval(j)) for (i, j) in re.findall(r, regular_str)])
        if "l2" in regular_str.lower():
            regular = L2Regularizer(**kwargs)
        else:
            raise ValueError("Unrecognized regular: {}".format(regular_str))
        return regular
