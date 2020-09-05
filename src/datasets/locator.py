import torch
import torchvision
from torchvision.transforms import transforms

from src.config.configs import Config
from src.datasets.closed_squares import ClosedSquaresDataset
from src.datasets.datasets import DatasetType, DatasetName
import numpy as np


class DatasetLocator:
    def __init__(self, conf: Config):

        self.dataset = conf.dataset
        self.gpu_run = conf.use_gpu
        self.batch_size = conf.batch_size
        train, valid, test = self.__load_data()

        self.dataset_dict = {
            DatasetType.TRAIN: train,
            DatasetType.VALID: valid,
            DatasetType.TEST: test
        }

    @staticmethod
    def __f(image):
        np_image = np.array(image)
        input_dim = np_image.shape[-1]
        new_image = np.zeros(shape=(60, 60), dtype=np.float32)
        i, j = np.random.randint(0, 60 - input_dim, size=2)
        new_image[i:i + input_dim, j:j + input_dim] = np_image
        return new_image

    def __transformed_mnist_transformation(self):
        return transforms.Compose(
            [torchvision.transforms.Lambda(self.__f),
             torchvision.transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))])

    @staticmethod
    def __augmented_mnist_transformation():
        return transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=(-180, 180), scale=(0.5, 1.0), ),
            torchvision.transforms.ToTensor()])

    @staticmethod
    def __augmented_mnist_simple_transformation():
        return transforms.Compose([
            torchvision.transforms.RandomAffine(degrees=(0, 90), scale=(0.9, 1.0), ),
            torchvision.transforms.ToTensor()])

    def __load_data(self):
        train_total = self.__load_dataset(True)
        test = self.__load_dataset(False)

        train_length = int(len(train_total) * 0.9)
        valid_length = len(train_total) - train_length
        (train, valid) = torch.utils.data.random_split(train_total, (train_length, valid_length))
        return train, valid, test

    def __load_dataset(self, is_train):
        transform = None
        if self.dataset == DatasetName.MNIST:
            transform = torchvision.transforms.ToTensor()
        elif self.dataset == DatasetName.AUGMENTED:
            transform = self.__augmented_mnist_transformation()
        elif self.dataset == DatasetName.TRANSFORMED:
            transform = self.__transformed_mnist_transformation()
        elif self.dataset == DatasetName.CLOSED_SQUARES:
            return ClosedSquaresDataset(train=is_train)
        return torchvision.datasets.MNIST(root='./data', train=is_train, download=True, transform=transform)

    def data_loader(self, dataset: DatasetType):
        should_shuffle = dataset == DatasetType.TRAIN
        data = self.dataset_dict[dataset]
        return torch.utils.data.DataLoader(data,
                                           batch_size=self.batch_size,
                                           pin_memory=self.gpu_run,
                                           shuffle=should_shuffle,
                                           num_workers=0)