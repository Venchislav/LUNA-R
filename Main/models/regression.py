# this code contains basic regression models
# LUNA-R is an educational project, meaning it can not be used in product development
# and is not as customizable as scikit-learn or something like this
# Code is under MIT license meaning it's free to share, copy, and use

import Main.configurations.cfg as cfg
from Main.tools.validator import validate_input
from Main.components.loss import mean_squared_error
from main_classes import Model

import numpy as np


class LinearRegression(Model):
    def __init__(self):
        super().__init__()
        self.sup_loss = cfg.supported_loss['linear_reg']

    def fit(self, X, y, epochs=100, lr=0.001, loss='mse'):
        n_samples, n_features = X.shape
        validate_input(X, y, epochs, lr, loss, self.sup_loss)  # check if everything is ok

        self.loss_fn = cfg.losses_addr[loss]
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(epochs):
            predictions = self.predict(X)
            self.loss_history.append(self.loss_fn(y, predictions))

            print(f'On Epoch {epoch} loss is {self.loss_fn(y, predictions)}')

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= 1 / n_samples * dw * lr
            self.bias -= 1 / n_samples * db * lr

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


# small test

# linear_model = LinearRegression()
# X = np.array([
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ])
# y = np.array([6, 15, 24])
#
# linear_model.fit(X, y, epochs=1000)
# print(linear_model.predict(X))
#
# print(linear_model.evaluate(X, y))
# print(linear_model.loss_history)
