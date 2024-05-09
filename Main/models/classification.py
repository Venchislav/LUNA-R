import Main.configurations.cfg as cfg

from Main.tools.validator import validate_input
from Main.tools.funcs import sigmoid
from main_classes import Model, Node

import numpy as np
from collections import Counter

# Logistic Regression
class LogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self.sup_loss = cfg.supported_loss['logistic_reg']

    def fit(self, y, X, epochs=100, lr=0.001, loss='binary_crossentropy'):
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


# DecisionTree Classification:

class DecisionTree:
    def __init__(self, max_depth=50, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.grow(X, y)

    def grow(self, X, y, cur_depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # base case
        if (cur_depth >= self.max_depth) or (n_samples < self.min_samples_split) or (n_labels == 1):
            val = self.most_common_category(y)
            return Node(value=val)

        feature_indices = np.random.choice(n_features, self.n_features, replace=True)  # n_features may not be equal to self.n_features
        split_feature, split_threshold = self.best_split(X, y, feature_indices)

        # left and right:

        left_indices, right_indices = self.split(X[:, split_feature], split_threshold)
        left = self.grow(X[left_indices, :], y[left_indices], cur_depth + 1)
        right = self.grow(X[right_indices, :], y[right_indices], cur_depth + 1)

        return Node(split_feature, split_threshold, left, right)

    def best_split(self, X, y, feature_indices):
        information_gain = -1
        split_feature_index, best_threshold = None, None

        for feature_index in feature_indices:
            X_col = X[:, feature_index]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                ig = self.information_gain(X_col, y, threshold)

                if ig > information_gain:
                    information_gain = ig
                    split_feature_index = feature_index
                    best_threshold = threshold
        return split_feature_index, best_threshold

    def information_gain(self, X_col, y, threshold):
        # parent entropy

        parent_e = self.entropy(y)

        # create_children
        left_indices, right_indices = self.split(X_col, threshold)

        # weighted entropy for children
        n = len(y)
        n_l, n_r = len(left_indices), len(right_indices)
        e_l, e_r = self.entropy(y[left_indices]), self.entropy(y[right_indices])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_e - child_entropy

    def entropy(self, y_values):
        p_s = np.bincount(y_values) / len(y_values)
        return -np.sum([p * np.log(p) for p in p_s if p > 0])

    def split(self, X_col, threshold):
        left_indices = np.argwhere(X_col <= threshold).flatten()
        right_indices = np.argwhere(X_col > threshold).flatten()

        return left_indices, right_indices

    def most_common_category(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        return [self.traverse_tree(x, self.root) for x in X]

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

# holy fucking shit...
