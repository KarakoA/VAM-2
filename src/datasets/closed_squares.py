from torch.utils.data import Dataset
import numpy as np
import torch
class ClosedSquaresDataset(Dataset):
    """Binary: number of not closed squares"""

    def __init__(self,
                 train,
                 size=64,
                 object_width=3,
                 n_missing=2,
                 n_classes=4,
                 n_circles=6,
                 total_train=16000,
                 total_test=1600):
        if train:
            np.random.seed(1)
        else:
            np.random.seed(2)
        self.n = total_train if train else total_test
        self.__create_data(n_classes, n_circles, size, object_width, n_missing)

    def __create_data(self, n_classes, n_circles, size, object_width, n_missing):
        self.labels = []
        self.data = []

        for class_i in range(n_classes):
            for _ in range(int(self.n / n_classes)):
                image = self.__generate_image(class_i, n_circles, size, object_width, n_missing)
                self.data.append(torch.tensor(image))
                self.labels.append(class_i)

    def __generate_image(self, n_open, n_all, size, object_width, n_missing):
        image = np.zeros((size, size))
        # top left x,y positions within bounds
        top_lefts = (np.random.rand(n_all, 2) * (size - (object_width + 2))).astype(int)
        # ensure no overlapping
        for top_left in top_lefts:
            x_0, y_0 = top_left[0], top_left[1]
            # 1 bigger so no overlaps
            image[x_0: x_0 + object_width + 2, y_0:y_0 + object_width + 2] += 1
        # make sure no overlapping
        is_valid = np.all(image <= 1)
        if is_valid:
            image = np.zeros((size, size)).astype(np.float32)
            for i, top_left in enumerate(top_lefts):
                x_0, y_0 = top_left[0] + 1, top_left[1] + 1
                image[x_0: x_0 + object_width, y_0:y_0 + object_width] = 1
                # open it
                if i < n_open:
                    pos = (np.random.rand(n_missing, 2) * object_width).astype(int)
                    for p in pos:
                        image[x_0 + p[0], y_0 + p[1]] = 0
            return image.reshape(1, size, size)
        else:
            return self.__generate_image(n_open, n_all, size, object_width, n_missing)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, uid):
        if torch.is_tensor(uid):
            uid = uid.tolist()
        label = self.labels[uid]
        sample = self.data[uid]

        return (sample, label)