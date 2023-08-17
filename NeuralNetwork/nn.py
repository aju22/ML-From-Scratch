import numpy as np
from layers import Dense, Activation

class NN:

    def __init__(self, features, lr=0.001):

        self.layers = []
        self.loss_fn = None
        self.loss_fn_prime = None

        for i in range(len(features)-1):
            self.layers.append(Dense(features[i], features[i + 1], lr))
            self.layers.append(Activation(activation='tanh'))

        self.layers.append(Activation(activation='softmax'))

    def show(self):
        for layer in self.layers:
            print(layer, end=" -> \n")

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def mse_prime(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    @staticmethod
    def binary_cross_entropy(y_true, y_pred):
        return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

    @staticmethod
    def binary_cross_entropy_prime(y_true, y_pred):
        return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    def predict(self, X):

        output = []
        for x in X:
            output.append(self.forward(x))

        return np.array(output)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, loss: str = "mse", verbose: bool = True):

        if loss == 'mse':
            loss_fn = self.mse
            loss_fn_prime = self.mse_prime

        elif loss == "bce":
            loss_fn = self.binary_cross_entropy
            loss_fn_prime = self.binary_cross_entropy_prime

        else:
            raise Exception("Loss Function not implemented")

        for e in range(epochs):

            error = 0

            for x, y in zip(X_train, y_train):

                # forward
                output = self.forward(x)

                # error
                error += loss_fn(y, output)

                # backward
                grad = loss_fn_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)

            error /= len(X_train)

            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")

            if error < 1e-5:
                break

