import numpy as np

eps = np.finfo(float).eps


class Loss:
    def value(y_true, y_pred):
        raise NotImplementedError

    def grad(y_true, y_pred):
        raise NotImplementedError


class MSE:
    pass


class BinaryCrossEntropy(Loss):

    @staticmethod
    def value(y_true, y_pred):
        return -np.sum(
            y_true * np.log(y_pred + eps) +
            (1 - y_true) * np.log(1 - y_pred + eps)
        )

    @staticmethod
    def grad(y_true, y_pred):
        NotImplementedError
