#!/usr/bin/python3
"""Logistic regression."""
from argparse import ArgumentParser
import numpy as np
import hw3_1a


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
    d1 = generate_data(args.n, *args.mean_var_pairs[:4])
    d2 = generate_data(args.n, *args.mean_var_pairs[-4:])


if __name__ == '__main__':
    main()
