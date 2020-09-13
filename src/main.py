# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# from datasets.datasets import DatasetName
import logging

from config.configs import Config
from datasets.datasets import DatasetType
from datasets.locator import DatasetLocator


import torch

from train.trainer import Trainer
from utils.utils import prepare_dirs


def main(config):
    prepare_dirs(config)

    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)

    locator = DatasetLocator(config)
    # instantiate data loaders
    if config.is_train:
        train_loader = locator.data_loader(DatasetType.TRAIN)
        valid_loader = locator.data_loader(DatasetType.VALID)
        dloader = (train_loader,valid_loader)
    else:
        dloader = locator.data_loader(DatasetType.TEST)

    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test()

if __name__ == '__main__':

    #logging.getLogger().setLevel(logging.DEBUG)
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format='%(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M')

    config = Config()
    main(config)
