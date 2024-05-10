from Main.models.tree import DecisionTreeClassifier
import numpy as np


class RandomForestClassifier:
    def __init__(self, n_trees, max_depth=50, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

        self.trees = []

    def fit(self, X, y):
        for t in range(self.n_trees):
            print(f'fitting tree {t + 1}/{self.n_trees}')
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          n_features=self.n_features)

            X_sampled, y_sampled = self.sample_with_replacement_data(X, y)

            tree.fit(X_sampled, y_sampled)
            self.trees.append(tree)

    def sample_with_replacement_data(self, X, y):
        samples = X.shape[0]
        random_indices = np.random.choice(samples, int(samples ** 0.5))  # by default replace is True

        return X[random_indices], y[random_indices]

    def most_frequent_label(self, y):
        counts = np.bincount(y)
        return np.argmax(counts)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        preds_for_tree = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self.most_frequent_label(pred) for pred in preds_for_tree])
        return predictions
