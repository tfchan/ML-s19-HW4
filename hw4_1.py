#!/usr/bin/python3
"""Logistic regression."""
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
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

    def predict(self, x):
        """Predict target using learnt coefficients."""
        x = self._preprocess(x)
        y = 1 / (1 + np.exp(-np.dot(x, self._coef)))
        y = (y > 0.5).astype(int)
        return y

    @property
    def coef(self):
        """Getter of model coefficients."""
        return self._coef


def generate_data(n, mean_x, var_x, mean_y, var_y):
    """Generate n (x, y) data points with corresponding mean and variance."""
    x = hw3_1a.normal(mean_x, var_x, n)
    y = hw3_1a.normal(mean_y, var_y, n)
    return np.vstack((x, y)).T


def plot(n_plot, title, position, x, y):
    """Plot subplot with specific title and position."""
    ax = plt.subplot(1, n_plot, position)
    ax.set_title(title)
    ax.plot(x[y == 0][:, 0], x[y == 0][:, 1], 'ro')
    ax.plot(x[y == 1][:, 0], x[y == 1][:, 1], 'bo')
    return ax


def perform_lg(methods, x, y):
    """Perform logistic regression with different methods."""
    n_plot = len(methods) + 1
    plt.figure()
    plot(n_plot, 'Groud truth', 1, x, y)
    method_count = 2
    for method_name, method in methods.items():
        method.fit(x, y)
        prediction = method.predict(x)
        print(f'{method_name}:\n')
        print(f'w:\n{method.coef}')
        plot(n_plot, method_name, method_count, x, prediction)
    plt.show()


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
    methods = {'Gradient descent': LogisticRegression(method='gradient')}
    perform_lg(methods, x, y)


if __name__ == '__main__':
    main()
