# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# from src.datasets.datasets import DatasetName
from src.config.configs import Config
from src.datasets.datasets import DatasetType
from src.datasets.locator import DatasetLocator


def run():
    conf = Config()
    locator = DatasetLocator(conf)
    dataset = locator.data_loader(DatasetType.TEST)

    print(iter(dataset).next())

# plot some imgs
# plot class distribution

if __name__ == '__main__':
    run()
