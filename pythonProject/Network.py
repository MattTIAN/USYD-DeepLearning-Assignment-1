import time
from collections import OrderedDict
from HiddenLayer import *
from Objective import *


def minibatch(X, batchsize=50, shuffle=True):
    """
    Split the dataset into batches and train based on mini batch
    """
    N = X.shape[0]
    idx = np.arange(N)
    n_batches = int(np.ceil(N / batchsize))
    if shuffle:
        np.random.shuffle(idx)

    def mb_generator():
        for i in range(n_batches):
            yield idx[i * batchsize: (i + 1) * batchsize]

    return mb_generator(), n_batches


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def early_stopping(valid):
    """
    :param valid：Validation set correctness list
    """
    if len(valid) > 5:
        if valid[-1] < valid[-5] and valid[-2] < valid[-5] and valid[-3] < valid[-5] and valid[-4] < valid[-5]:
            return True
    return False


class DFN(object):

    def __init__(
            self,
            hidden_dims_1=None,
            hidden_dims_2=None,
            optimizer="sgd(lr=0.01)",
            init_w="std_normal",
            regular_act=None,
            p=None,
            BN=None,
            loss=CrossEntropy()):
        self.p = p
        self.BN = BN
        self.optimizer = optimizer
        self.init_w = init_w
        self.loss = loss
        self.regular_act = regular_act
        self.regular = None
        self.hidden_dims_1 = hidden_dims_1
        self.hidden_dims_2 = hidden_dims_2
        self.is_initialized = False

    def _set_params(self):
        """
        FC1 -> relu -> FC2 -> relu
        """
        self.layers = OrderedDict()
        self.layers["FC1"] = FullyConnected(
            n_out=self.hidden_dims_1,
            acti_fn="relu",
            init_w=self.init_w,
            optimizer=self.optimizer,
            p=self.p,
            BN=self.BN
        )
        self.layers["FC2"] = FullyConnected(
            n_out=self.hidden_dims_2,
            acti_fn="relu",
            init_w=self.init_w,
            optimizer=self.optimizer,
            p=self.p,
            BN=self.BN
        )

        if self.regular_act is not None:
            self.regular = RegularizerInitializer(self.regular_act)()
        self.is_initialized = True

    def forward(self, X_train, is_train=True, BN='train'):
        Xs = {}
        out = X_train
        for k, v in self.layers.items():
            Xs[k] = out
            out = v.forward(out, BN)
        return out, Xs

    def backward(self, grad):
        dXs = {}
        out = grad
        for k, v in reversed(list(self.layers.items())):
            dXs[k] = out
            out = v.backward(out, regular=self.regular)
        return out, dXs

    def update(self):
        """
        Gradient update
        """
        for k, v in reversed(list(self.layers.items())):
            v.update()
        self.flush_gradients()

    def flush_gradients(self, curr_loss=None):
        """
        Reset gradient after update
        """
        for k, v in self.layers.items():
            v.flush_gradients()

    def fit(self, X_train, y_train, n_epochs=20, batch_size=64, verbose=False):
        """
        :param X_train：Training data
        :param y_train：Training Data Labeling
        :param n_epochs：epoch Number of times
        :param batch_size：batch size per epoch
        :param verbose：Whether the output loss per batch
        """
        self.verbose = verbose
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        if not self.is_initialized:
            self.n_features = X_train.shape[1]
            self._set_params()
        prev_loss = np.inf
        train_acc = []
        for i in range(n_epochs):
            acc = 0.0
            loss, epoch_start = 0.0, time.time()
            batch_generator, n_batch = minibatch(X_train, self.batch_size, shuffle=True)

            for j, batch_idx in enumerate(batch_generator):
                batch_len, batch_start = len(batch_idx), time.time()
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                out, _ = self.forward(X_batch)
                y_pred_batch = softmax(out)
                batch_loss = self.loss(y_batch, y_pred_batch)

                # Record the accumulated accuracy in each batch
                y_train_pred = np.argmax(y_pred_batch, axis=1)
                y_train_batch = np.argmax(y_batch, axis=1)
                acc += np.sum(y_train_pred == y_train_batch)

                # Regularization loss
                if self.regular is not None:
                    for _, layerparams in self.hyperparams['components'].items():
                        assert type(layerparams) is dict
                        batch_loss += self.regular.loss(layerparams)

                grad = self.loss.grad(y_batch, y_pred_batch)
                _, _ = self.backward(grad)
                self.update()
                loss += batch_loss
                if self.verbose:
                    fstr = "\t[Batch {}/{}] Train loss: {:.3f} ({:.1f}s/batch)"
                    print(fstr.format(j + 1, n_batch, batch_loss, time.time() - batch_start))
            loss /= n_batch
            acc /= X_train.shape[0]
            train_acc.append(acc)
            fstr = "[Epoch {}] Avg. loss: {:.3f} Delta: {:.3f} ({:.2f}m/epoch) Training Accuracy: {:.4f}"
            print(fstr.format(i + 1, loss, prev_loss - loss, (time.time() - epoch_start) / 60.0, acc))
            prev_loss = loss
        return train_acc

    def evaluate(self, X_test, y_test, BN='test', batch_size=128, verbose=False):
        acc = 0.0
        confusion_matrix = np.zeros((10, 10))
        batch_generator, n_batch = minibatch(X_test, batch_size, shuffle=True)
        for j, batch_idx in enumerate(batch_generator):
            batch_len, batch_start = len(batch_idx), time.time()
            X_batch, y_batch = X_test[batch_idx], y_test[batch_idx]
            y_pred_batch, _ = self.forward(X_batch, BN)
            y_pred_batch = np.argmax(y_pred_batch, axis=1)
            y_batch = np.argmax(y_batch, axis=1)
            acc += np.sum(y_pred_batch == y_batch)
            for t_ in y_batch:
                for p_ in y_pred_batch:
                    confusion_matrix[t_][p_] += 1
        return acc / X_test.shape[0], confusion_matrix

    @property
    def hyperparams(self):
        return {"init_w": self.init_w,
                "loss": str(self.loss),
                "optimizer": self.optimizer,
                "regular": str(self.regular_act),
                "hidden_dims_1": self.hidden_dims_1,
                "hidden_dims_2": self.hidden_dims_2,
                "dropout keep ratio": self.p,
                "components": {k: v.params for k, v in self.layers.items()}
                }
