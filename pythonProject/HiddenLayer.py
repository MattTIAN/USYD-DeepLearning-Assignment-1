from Initializer import *


class LayerBase(ABC):
    def __init__(self, optimizer=None):
        self.X = []  # Network layer input
        self.gradients = {}  # Network layer pending gradient update variables
        self.params = {}  # Network layer parameter variables
        self.acti_fn = None  # Network layer activation function
        self.optimizer = OptimizerInitializer(optimizer)()  # Network layer optimization methods

    @abstractmethod
    def _init_params(self, **kwargs):
        """
        Initialize parameters
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, X, **kwargs):
        """
        Forward Propagation
        """

        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """
        Back Propagation
        """
        raise NotImplementedError

    def flush_gradients(self):
        """
        Reset the list of update parameters
        """
        self.X = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self):
        """
        Update parameters
        """

        for k, v in self.gradients.items():
            if k in self.params:
                self.params[k] = self.optimizer(self.params[k], v, k)


"""
BN Algorithm
"""


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == 'train':

        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        out_ = (x - sample_mean) / np.sqrt(sample_var + eps)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        out = gamma * out_ + beta
        cache = (out_, x, sample_var, sample_mean, eps, gamma, beta)

    elif mode == 'test':

        scale = gamma / np.sqrt(running_var + eps)
        out = x * scale + (beta - running_mean * scale)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    out_, x, sample_var, sample_mean, eps, gamma, beta = cache

    N = x.shape[0]
    dout_ = gamma * dout
    dvar = np.sum(dout_ * (x - sample_mean) * -0.5 * (sample_var + eps) ** -1.5, axis=0)
    dx_ = 1 / np.sqrt(sample_var + eps)
    dvar_ = 2 * (x - sample_mean) / N

    # intermediate for convenient calculation
    di = dout_ * dx_ + dvar * dvar_
    dmean = -1 * np.sum(di, axis=0)
    dmean_ = np.ones_like(x) / N

    dx = di + dmean * dmean_
    dgamma = np.sum(dout * out_, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta


class FullyConnected(LayerBase):
    """
    Define the fully connected layer, implementing a=g(x*W+b),
    forward propagating the input x, returning a;
    backward propagating the input
    """

    def __init__(self, n_out, acti_fn, init_w, optimizer=None, p=None, BN=None):
        """
        :param acti_fn：Activation function, str type
        :param init_w：Weight initialization method, str type
        :param n_out：Hidden layer output dimension
        :param optimizer：Optimization methods
        :param p：Dropout retention rate
        :param BN：Batch normalization switch
        """
        super().__init__(optimizer)
        self.n_in = None  # Hidden layer input dimension, int type
        self.n_out = n_out  # Output dimension of hidden layer, int type
        self.acti_fn = ActivationInitializer(acti_fn)()
        self.init_w = init_w
        self.init_weights = WeightInitializer(mode=init_w)
        self.is_initialized = False  # Initialized or not, bool variable
        self.p = p
        self.BN = BN

    def _init_params(self):
        b = np.zeros((1, self.n_out))
        W = self.init_weights((self.n_in, self.n_out))
        self.params = {"W": W, "b": b}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.is_initialized = True
        self.mask = None
        self.bn_param = {'mode': self.BN}
        self.cache = ()
        self.beta = 1.0
        self.gamma = 1.0

    def forward(self, X, retain_derived=True, BN='train'):
        """
        Forward propagation of fully connected networks
        :param X：Input array of (n_samples, n_in), float type
        :param retain_derived：Whether to keep intermediate variables for reuse in backpropagation, bool type
        """
        if not self.is_initialized:  # Initialize the parameters first if they are not initialized
            self.n_in = X.shape[1]
            self._init_params()
        mask = np.ones(X.shape).astype(bool)
        W = self.params["W"]
        b = self.params["b"]
        z = X @ W + b

        """
        If parameter BN exists, BN conducts
        """
        if self.BN is not None:
            if BN == 'test':
                self.bn_param['mode'] = 'test'
            z, self.cache = batchnorm_forward(z, self.gamma, self.beta, self.bn_param)
        a = self.acti_fn.forward(z)
        """
        If parameter p exists, dropout conducts
        """
        if self.p is not None:
            mask = (np.random.rand(*a.shape) < self.p) / self.p
            a = mask * a
        if retain_derived:
            self.X.append(X)
        return a

    def backward(self, dLda, retain_grads=True, regular=None):
        """
        Back propagation of fully connected networks
        :param dLda：On the gradient of the loss, for (n_samples, n_out), float type
        :param retain_grads：Whether to calculate the parameter gradient of the intermediate variable, bool type
        :param regular：Regularization term
        """

        if not isinstance(dLda, list):
            dLda = [dLda]
        dX = []
        X = self.X
        for da, x in zip(dLda, X):
            dx, dw, db = self._bwd(da, x, regular)
            dX.append(dx)
            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db
        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLda, X, regular):
        W = self.params["W"]
        b = self.params["b"]
        Z = X @ W + b
        """
        If parameter BN exits, BN back propagation
        """
        if self.BN is not None:
            Z, self.cache = batchnorm_forward(Z, self.gamma, self.beta, self.bn_param)
        dZ = dLda * self.acti_fn.grad(Z)
        """
        If parameter BN exits, BN back propagation
        """
        if self.BN is not None:
            dZ, dgamma, dbeta = batchnorm_backward(dZ, self.cache)
        dX = dZ @ W.T
        dW = X.T @ dZ
        db = dZ.sum(axis=0, keepdims=True)
        if regular is not None:
            n = X.shape[0]
            dW_norm = regular.grad(self.params) / n
            dW += dW_norm
        return dX, dW, db

    @property
    def hyperparams(self):
        return {"layer": "FullyConnected", "init_w": self.init_w,
                "n_in": self.n_in,
                "n_out": self.n_out,
                "acti_fn": str(self.acti_fn),
                "optimizer": {
                    "hyperparams": self.optimizer.hyperparams,
                },
                "components": {
                    k: v for k, v in self.params.items()
                }
                }


class Softmax(LayerBase):
    """
    Define Softmax layer
    """

    def __init__(self, dim=-1, optimizer=None):

        super().__init__(optimizer)
        self.dim = dim
        self.n_in = None
        self.is_initialized = False

    def _init_params(self):

        self.params = {}
        self.gradients = {}
        self.is_initialized = True

    def forward(self, X, retain_derived=True):

        """
        Forward propagation of Softmax
        """
        if not self.is_initialized:
            self.n_in = X.shape[1]
            self._init_params()
        Y = self._fwd(X)
        if retain_derived:
            self.X.append(X)
        return Y

    def _fwd(self, X):

        e_X = np.exp(X - np.max(X, axis=self.dim, keepdims=True))
        return e_X / e_X.sum(axis=self.dim, keepdims=True)

    def backward(self, dLdy):

        """
        Backpropagation of Softmax
        """
        if not isinstance(dLdy, list):
            dLdy = [dLdy]
        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx = self._bwd(dy, x)
            dX.append(dx)
        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):
        dX = []
        for dy, x in zip(dLdy, X):
            dxi = []
            for dyi, xi in zip(*np.atleast_2d(dy, x)):
                yi = self._fwd(xi.reshape(1, -1)).reshape(-1, 1)
                dyidxi = np.diagflat(yi) - yi @ yi.T
                dxi.append(dyi @ dyidxi)
            dX.append(dxi)
        return np.array(dX).reshape(*X.shape)

    @property
    def hyperparams(self):
        return {"layer": "SoftmaxLayer", "n_in": self.n_in,
                "n_out": self.n_in,
                "optimizer": {
                    "hyperparams": self.optimizer.hyperparams,
                },
                }
