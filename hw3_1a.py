#!python3
"""Univariate Gaussian data generator."""
from argparse import ArgumentParser
import numpy as np


def std_normal(n=1):
    """Generate n standart normal distributed values."""
    u = np.random.uniform(size=n)
    v = np.random.uniform(size=n)
    std_norm = (-2 * np.log(u))**0.5 * np.cos(2 * np.pi * v)
    return std_norm


def normal(mean=0.0, var=1.0, n=1):
    """Generate n normal distributed values."""
    std_dev = var**0.5
    return mean + std_dev * std_normal(n)


def main():
    """Perform main task of the program."""
    parser = ArgumentParser(description='Univariate Gaussian data generator')
    parser.add_argument('mean', type=float, help='Gaussian mean')
    parser.add_argument('variance', type=float, help='Gaussian variance')
    parser.add_argument('-n', '--n_data', type=int, default=1,
                        help='Number of data to generate')
    args = parser.parse_args()

    values = normal(args.mean, args.variance, args.n_data)
    print(*values, sep='\n')


if __name__ == '__main__':
    main()
