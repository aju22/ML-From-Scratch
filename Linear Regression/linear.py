import numpy as np

class LinearRegression:

    def __init__(self, lr=0.001, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.w = 0
        self.b = 0

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for i in range(self.max_iter):

            predictions = self.predict(X)
            self.w += self.lr * (2 / len(X)) * np.dot(X.T, (y - predictions))
            self.b += self.lr * (2 / len(X)) * np.sum(y - predictions)

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def score(self, X, y):
        accuracy = np.sum(self.predict(X) == y) / len(y)

    def get_params(self, deep=False):
        return {"w": self.w, "b": self.b}