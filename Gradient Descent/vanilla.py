import numpy as np
import matplotlib.pyplot as plt

class Optimizers:

    def __init__(self, fn, derivative, lr = 0.001):
        self.fn = fn
        self.derivative = derivative
        self.lr = lr
        self.x = np.arange(-100, 100, 0.1)
        self.y = self.fn(self.x)

    def vanilla(self, animate=False):

        error = np.inf
        n_iters = 0
        point_x = self.x[0]

        while abs(error) > 0.0001:

            point_x = point_x - self.lr * self.derivative(point_x)
            point_y = self.fn(point_x)

            error = 0 - point_y
            n_iters += 1

            if animate:
                self.plot(point_x, point_y)

        print("No. of iterations for Vanilla Gradient Descent: ", n_iters)


    def momentum(self, gamma = 0.8, animate = False):

        error = np.inf
        n_iters = 0
        point_x = self.x[0]
        update = 0

        while abs(error) > 0.0001:

            update = gamma * update + self.lr * self.derivative(point_x) # Momentum Update Rule

            point_x = point_x - update
            point_y = self.fn(point_x)

            error = 0 - point_y
            n_iters += 1

            if animate:
                self.plot(point_x, point_y)

        print("No. of iterations for Momentum Gradient Descent: ", n_iters)



    def plot(self, x, y):
        plt.plot(self.x, self.y)
        plt.scatter(x, y, c='red')
        plt.pause(0.00001)
        plt.clf()


def fn(x):
    return x**2

def derivative(x):
    return 2.0*x

opt = Optimizers(fn, derivative, lr = 0.01)

#opt.vanilla(animate=True)
opt.momentum(animate=True)











