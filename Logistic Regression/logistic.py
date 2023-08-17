import numpy as np

class LogisticRegression:

    def __init__(self, lr=0.001, max_iters=10000):
        self.lr = lr
        self.max_iters = max_iters

        self.weights = None
        self.bias = 0

    @staticmethod
    def sigmoid(X):
        return 1.0 / (1 + np.exp(-1 * X))

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iters):
            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)

            self.weights -= self.lr * (2/n_samples) * (np.dot(X.T, (predictions-y)))
            self.bias -= self.lr * (2/n_samples) * np.sum(predictions - y)

    def predict(self, X, thresh = 0.5):

        linear = np.dot(X, self.weights) + self.bias
        logits = self.sigmoid(linear)

        return np.array([0 if pred <= thresh else 1 for pred in logits])

    @staticmethod
    def accuracy(y_pred, y_test):
        return np.sum(y_pred == y_test)/len(y_test)

    def get_param(self):
        return {"w": self.weights, "b": self.bias}