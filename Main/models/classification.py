import Main.configurations.cfg as cfg

from Main.tools.validator import validate_input
from Main.tools.funcs import sigmoid
from main_classes import Model

import numpy as np


class LogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self.sup_loss = cfg.supported_loss['logistic_reg']

    def fit(self, X, y, epochs=100, lr=0.001, loss='binary_crossentropy'):
        n_samples, n_features = X.shape
        validate_input(X, y, epochs, lr, loss, self.sup_loss)  # check if everything is ok
        self.loss_fn = cfg.losses_addr[loss]

        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):
            predictions = sigmoid(np.dot(X, self.weights) + self.bias)
            self.loss_history.append(self.loss_fn(y, predictions))

            print(f'On Epoch {epoch} loss is {cfg.losses_addr[loss](y, predictions)}')

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= dw * lr
            self.bias -= db * lr

    def predict(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)
