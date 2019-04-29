#!/usr/bin/python3
"""Logistic regression."""
from argparse import ArgumentParser


def main():
    """Do main task."""
    parser = ArgumentParser(description="Logistic regression\
        using Newton's or gradient descent method")
    parser.add_argument('n', type=int, help='Number of data points to generate\
        for regression')
    parser.add_argument('mean_var_pairs', type=float, nargs=8,
                        help='Mean and variance pairs for\
                        4 data point generators')
    args = parser.parse_args()


if __name__ == '__main__':
    main()
