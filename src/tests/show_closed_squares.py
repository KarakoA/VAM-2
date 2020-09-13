from datasets.closed_squares import ClosedSquaresDataset

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

from utils import plots


def run():
    test = ClosedSquaresDataset(train=False)
    train = ClosedSquaresDataset(train=True)

    print(f"Test set length: {len(test)} entires")
    print(f"Train set length: {len(train)} entries")

    print("some images...")
    images, labels = test[0:3]
    plots.show_images(images, labels)

    images, labels = test[100:103]
    plots.show_images(images, labels)

    images, labels = test[200:203]
    plots.show_images(images, labels)

    images, labels = test[300:303]
    plots.show_images(images, labels)

    print("Per class distributions: ")
    print(f"test: {np.bincount(test.labels)}")
    print(f"train: {np.bincount(train.labels)}")


if __name__ == '__main__':
    run()
