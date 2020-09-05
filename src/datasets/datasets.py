from enum import Enum

class DatasetType(Enum):
    TRAIN = 1
    VALID = 2
    TEST  = 3

class DatasetName(Enum):
    MNIST = 1
    AUGMENTED = 2
    TRANSFORMED = 3
    AUGMENTED_MEDICAL = 4
    CLOSED_SQUARES = 5
