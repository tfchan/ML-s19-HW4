#!/usr/bin/python3
"""EM algorithm on MINST."""
import argparse
import os
import struct
import numpy as np


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
    return imgs.reshape((imgs.shape[0], imgs.shape[1] * imgs.shape[2]))


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

    tr_x = imgs2features(args.tr_image)
    ts_x = imgs2features(args.ts_image)
    tr_y = args.tr_label
    ts_y = args.ts_label


if __name__ == '__main__':
    main()
