import numpy as np
from collections import Counter


class KNN:

    def __init__(self, k=3):
        self.y_train = None
        self.X_train = None
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self.predict_(x) for x in X])

    def predict_(self, x):
        distances = np.array([np.linalg.norm(x - x_train) for x_train in self.X_train])
        k_indices = np.argsort(distances)[:self.k]
        y_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(y_labels).most_common(1)[0][0]

        return most_common
