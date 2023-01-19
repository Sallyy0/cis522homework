import numpy as np


class LinearRegression:
    """
    A simple linear regression model
    """

    w: np.ndarray
    b: float

    def __init__(self):
        self.w = np.zeros(2)
        self.b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        fitting a linear regression using closed form
        """
        self.w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        predicting output using closed form linear regression

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.
        """
        pred = np.dot(X, self.w) + self.b
        return pred


class GradientDescentLinearRegression(LinearRegression):
    """A linear regression model that uses gradient descent to fit the model."""

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        """fitting a linear regression model by using gradient descent"""
        self.w = np.zeros(len(X[0]))
        for i in range(epochs):
            y_pred = X.dot(self.w) + self.b
            dW = (-2 / len(X)) * sum(X.dot((y - y_pred)))
            db = (-1 / len(X)) * sum(y - y_pred)
            self.w = self.w - lr * dW
            self.b = self.b - lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output for the given input.

        Arguments:
            X (np.ndarray): The input data.

        Returns:
            np.ndarray: The predicted output.

        """
        pred = X.dot(self.w) + self.b
        return pred
