#!/usr/bin/python3
"""Logistic regression."""
from argparse import ArgumentParser
import numpy as np
import hw3_1a


class LogisticRegression:
    """Class for doing logistic regression."""

    methods = {'gradient': '_fit_gradient', 'newtons': '_fit_newtons'}

    def __init__(self, method='gradient'):
        """Initialize with specific method."""
        if method not in self.methods.keys():
            msg = (f'No {method} method, '
                   + f'choose from {list(self.methods.keys())}')
            raise Exception(msg)
        self._method = method
        self._coef = None

    @staticmethod
    def _preprocess(x):
        """Preprocess x."""
        return np.hstack((x, np.ones((x.shape[0], 1))))

    def fit(self, x, y):
        """Fit incoming data using chosen method."""
        x = self._preprocess(x)
        self._coef = np.random.rand(x.shape[1])
        getattr(self, self.methods.get(self._method))(x, y)

    def _fit_gradient(self, x, y):
        """Fit incoming data using gradient descent."""
        converge = False
        while not converge:
            logistic = 1 / (1 + np.exp(-np.dot(x, self._coef)))
            step = x.T @ (y - logistic)
            self._coef = self._coef + step
            converge = (np.absolute(step) < 0.0001).all()


def generate_data(n, mean_x, var_x, mean_y, var_y):
    """Generate n (x, y) data points with corresponding mean and variance."""
    x = hw3_1a.normal(mean_x, var_x, n)
    y = hw3_1a.normal(mean_y, var_y, n)
    return np.vstack((x, y)).T


def main():
    """Do main task."""
    # Parse arguments
    parser = ArgumentParser(description="Logistic regression\
        using Newton's or gradient descent method")
    parser.add_argument('n', type=int, help='Number of data points to generate\
        for regression')
    parser.add_argument('mean_var_pairs', type=float, nargs=8,
                        help='Mean and variance pairs for\
                        4 data point generators')
    args = parser.parse_args()

    # Generate data points
    x1 = generate_data(args.n, *args.mean_var_pairs[:4])
    y1 = np.zeros((args.n))
    x2 = generate_data(args.n, *args.mean_var_pairs[-4:])
    y2 = np.ones((args.n))
    x = np.concatenate((x1, x2))
    y = np.concatenate((y1, y2))

    # Train model
    gradient_lg = LogisticRegression(method='gradient')
    gradient_lg.fit(x, y)


if __name__ == '__main__':
    main()
