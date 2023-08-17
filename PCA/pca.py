import numpy as np

class PCA:

    def __init__(self, n_components):
        self.components = None
        self.n_components = n_components
        self.mean = None

    def fit(self, X):

        self.mean = np.mean(X, axis=0)
        X -= self.mean

        cov = np.cov(X.T)

        eigenvectors, eigenvalues  = np.linalg.eig(cov)

        eigenvectors = eigenvectors.T

        max_eigenvalues_idx = np.argsort(eigenvalues)[::-1]
        max_eigenvectors = eigenvectors[max_eigenvalues_idx]

        max_k_eigenvectors = max_eigenvectors[:self.n_components]

        self.components = max_k_eigenvectors

    def transform(self, X):
        X -= self.mean
        return np.dot(X, self.components.T)
