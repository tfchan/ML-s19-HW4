#!/usr/bin/python3
"""EM algorithm on MINST."""
import argparse
import os
import struct
import numpy as np
import matplotlib.pyplot as plt


def mnist_data(path):
    """Read mnist file from path and return data with appropriate shape."""
    # Test if file exist
    if not os.path.exists(path):
        msg = f'{path} does not exist, use `-h` for instructions'
        raise argparse.ArgumentTypeError(msg)

    # Try to open and read MNIST data from file
    with open(path, 'rb') as f:
        try:
            dimension = struct.unpack_from('>B', f.read(4), 3)[0]
            shape = []
            for _ in range(dimension):
                shape += [struct.unpack('>I', f.read(4))[0]]
            data = np.fromfile(f, dtype='uint8').reshape(tuple(shape))
        except struct.error:
            msg = f'{path} is not a valid MNIST file'
            raise argparse.ArgumentTypeError(msg)
    return data


def imgs2features(imgs):
    """Convert array of images to array of features."""
    imgs = (imgs >= 128).astype(int)
    return imgs.reshape((imgs.shape[0], imgs.shape[1] * imgs.shape[2]))


def log_(x):
    """Do log depends on some special condition."""
    return np.log(np.where(x > 1e-50, x, 1e-50))


def em_algorithm(x, y, n_cluster=2):
    """Perfrom EM on data x with n_cluster."""
    # Initialize parameters
    n_sample = x.shape[0]
    n_feature = x.shape[1]
    cluster_proba = np.random.random((n_cluster))
    cluster_proba /= cluster_proba.sum(axis=0)
    black_proba = np.random.random((n_cluster, n_feature))

    converge = False
    iter = 0
    while not converge:
        print(f'Iteration {iter}...\r', end='')
        # E step
        w = (np.dot(x, log_(black_proba.T))
             + np.dot(1 - x, log_(1 - black_proba.T))
             + log_(cluster_proba))
        w = w - w.max(axis=0)
        w = np.exp(w)
        w = w / w.sum(axis=0)

        # M step
        new_cluster_proba = w.sum(axis=0) / n_sample
        new_black_proba = ((w.T @ x) / w.sum(axis=0)[:, None])

        # Determine convergence
        converge = ((np.abs(new_cluster_proba - cluster_proba) < 1e-10).all()
                    and (np.abs(new_black_proba - black_proba) < 1e-10).all())
        cluster_proba = new_cluster_proba
        black_proba = new_black_proba
        iter += 1
    print()

    # Get cluster of each image
    result = (np.dot(x, log_(black_proba.T))
              + np.dot(1 - x, log_(1 - black_proba.T))
              + log_(cluster_proba))
    prediction = np.argmax(result, axis=1)
    pred_y = np.zeros(n_sample)
    for j in range(n_cluster):
        c_j_idx = [idx for idx, pred in enumerate(prediction) if pred == j]
        c_j_label = [y[k] for k in c_j_idx]
        label, count = np.unique(c_j_label, return_counts=True)
        pred_y[c_j_idx] = count.argmax()

    # Output result
    n_correct = 0
    for j in range(n_cluster):
        tp = ((pred_y == j) & (y == j)).sum()
        tn = ((pred_y != j) & (y != j)).sum()
        fp = ((pred_y == j) & (y != j)).sum()
        fn = ((pred_y != j) & (y == j)).sum()
        n_correct += tp
        print(f'Confusion matrix {j}:')
        print(f'\t\tPredict number {j}\tPredict not number {j}')
        print(f'Is number {j}\t\t{tp}\t\t\t{fn}')
        print(f"Isn't number {j}\t\t{fp}\t\t\t{tn}\n")
        print(f'Sensitivity (Successfully predict cluster 1):',
              tp / (tp + fn))
        print(f'Specificity (Successfully predict cluster 2):',
              tn / (tn + fp))
        print('\n------------------------------------------------------------')
    print(f'Total iteration to converge: {iter}')
    print(f'Total error rate: {1 - (n_correct / n_sample)}')

    # Plot cluster imaginations
    plt.figure()
    for c in range(n_cluster):
        plt.subplot(2, 5, c + 1)
        plt.imshow((black_proba[c] > 0.5).reshape((28, 28)))
    plt.show()


def main():
    """Do main task."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='EM algorithm on MNIST')
    parser.add_argument('--tr_image', type=mnist_data,
                        default='data/train-images.idx3-ubyte',
                        help='Path to MNIST training image data,\
                        default data/train-images.idx3-ubyte')
    parser.add_argument('--tr_label', type=mnist_data,
                        default='data/train-labels.idx1-ubyte',
                        help='Path to MNIST training label data,\
                        default data/train-labels.idx1-ubyte')
    parser.add_argument('--ts_image', type=mnist_data,
                        default='data/t10k-images.idx3-ubyte',
                        help='Path to MNIST testing image data,\
                        default data/t10k-images.idx3-ubyte')
    parser.add_argument('--ts_label', type=mnist_data,
                        default='data/t10k-labels.idx1-ubyte',
                        help='Path to MNIST testing label data,\
                        default data/t10k-labels.idx1-ubyte')
    args = parser.parse_args()

    # Convert data
    tr_x = imgs2features(args.tr_image)
    # ts_x = imgs2features(args.ts_image)
    tr_y = args.tr_label
    # ts_y = args.ts_label

    # Perform EM
    em_algorithm(tr_x, tr_y, n_cluster=10)


if __name__ == '__main__':
    main()
