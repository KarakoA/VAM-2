import torch
import numpy as np
import matplotlib.pyplot as plt

from src.config.configs import Config
from src.datasets.datasets import DatasetType
from src.datasets.locator import DatasetLocator
from src.train.trainer import Trainer

def run():
    config = Config()
    config.resume = True
    config.is_train = False

    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)

    locator = DatasetLocator(config)
    # instantiate data loaders
    dloader = locator.data_loader(DatasetType.TEST)

    trainer = Trainer(config, dloader)

    predictions = trainer.test().numpy()

    labels = np.array(locator.dataset_dict[DatasetType.TEST].labels)
    correct = labels == predictions

    correct_counts = np.bincount(labels[correct])

    wrong = labels != predictions
    wrong_counts = np.bincount(labels[wrong])

    class_labels = ['0', '1', '2', '3']
    width = 0.35
    fig, ax = plt.subplots()

    ax.bar(class_labels, correct_counts, width, label='Correct', color="b")
    ax.bar(class_labels, wrong_counts, width, bottom=correct_counts, label='Wrong', color="r")
    ax.set_ylabel('N')
    ax.set_title('Predictions per number of cells (classes)')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    run()