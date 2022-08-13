import numpy as np


class std_normal:
    """
    Standard normal initialization
    """

    def __init__(self, gain=0.01):
        self.gain = gain

    def __call__(self, weight_shape):
        return self.gain * np.random.randn(*weight_shape)

