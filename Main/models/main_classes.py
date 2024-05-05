class Model:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.loss_fn = None

        self.loss_history = []

    def fit(self, loss):
        self.loss_history.append(loss)

    def predict(self):
        pass

    def evaluate(self, X, y):
        return self.loss_fn(self.predict(X), y)  # don't worry. It works in subclasses
