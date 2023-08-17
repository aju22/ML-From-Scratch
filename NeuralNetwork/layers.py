import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        pass


class Dense(Layer):

    def __init__(self, input_dim, output_dim, lr=0.001):
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.lr: float = lr
        self.weights: np.ndarray = np.random.randn(output_dim, input_dim)
        self.bias: np.ndarray = np.random.randn(output_dim, 1)
        self.input: np.ndarray = np.empty((input_dim, ))
        self.output: np.ndarray = np.empty((output_dim, ))

    def forward(self, x: np.ndarray) -> np.ndarray:

        assert x.shape[0] == self.input_dim, f"Dimension Mismatch: X->{x.shape} not matching ({self.input_dim},)"
        self.input = x
        self.output = np.dot(self.weights, x) + self.bias

        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:

        input_gradient = np.dot(self.weights.T, output_gradient)
        weight_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = output_gradient

        self.weights -= self.lr * weight_gradient
        self.bias -= self.lr * bias_gradient

        return input_gradient

    def __repr__(self):

        return f"Dense({self.input_dim},{self.output_dim})"


class Activation(Layer):

    def __init__(self, activation: str = 'tanh'):
        self.activation =  None
        self.activation_prime = None
        self.input = None
        self.output = None
        self.activation_name = activation

        if activation == 'tanh':
            self.activation = lambda x: np.tanh(x)
            self.activation_prime = lambda x: 1 - (np.tanh(x))**2

        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
            self.activation_prime = lambda x: np.where(x > 0, 1, 0)

        elif activation == 'softmax':
            self.activation = lambda x: np.exp(x - np.max(x))/(np.exp(x - np.max(x))).sum(axis=0)
            self.activation_prime = lambda x: self.activation(x) * (1 - self.activation(x))

        else:
            raise Exception("Activation Function not Implemented.")

    def forward(self, x: np.ndarray) -> np.ndarray:

        self.input = x
        self.output = self.activation(x)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:

        return np.multiply(output_gradient, self.activation_prime(self.input))

    def __repr__(self):

       return f"Activation(activation={self.activation_name})"