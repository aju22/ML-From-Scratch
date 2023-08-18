import numpy as np

class PCA:

    def __init__(self, n_components):
        """
        Initialize PCA with the desired number of principal components.

        Args:
        n_components (int): The number of principal components to retain.
        """
        self.components = None
        self.n_components = n_components
        self.mean = None

    def fit(self, X):
        """
        Fit PCA on the input data.

        Args:
        X (numpy.ndarray): Input data matrix with shape (n_samples, n_features).
        """
        # Compute the mean of each feature across all samples
        self.mean = np.mean(X, axis=0)

        # Center the data by subtracting the mean from each feature
        X -= self.mean

        # Compute the covariance matrix of the centered data
        cov = np.cov(X.T)

        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        # Transpose the eigenvectors matrix to have one eigenvector per column
        eigenvectors = eigenvectors.T

        # Sort eigenvalues in descending order and get the corresponding indices
        max_eigenvalues_idx = np.argsort(eigenvalues)[::-1]

        # Reorder the eigenvectors based on the sorted eigenvalues
        max_eigenvectors = eigenvectors[max_eigenvalues_idx]

        # Select the top k eigenvectors as principal components
        max_k_eigenvectors = max_eigenvectors[:self.n_components]

        # Store the selected principal components
        self.components = max_k_eigenvectors

    def transform(self, X):
        """
        Transform the input data into the new reduced-dimensional space.

        Args:
        X (numpy.ndarray): Input data matrix with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Transformed data in the reduced-dimensional space.
        """
        # Center the data by subtracting the mean from each feature
        X -= self.mean

        # Project the centered data onto the principal components
        transformed_data = np.dot(X, self.components.T)

        return transformed_data
