from torch.utils.data import Dataset
import numpy as np
import torch
import os
import torchvision
class AugmentedMedicalMNISTDataset(Dataset):
    """
    Augmented mnist meant to mimic whole-slide-images of tumor cells.
    9's represent cancer cells. There are 4 different labels, based on the number of 9's:

    zero 9's          - no cancer
    one 9             - isolated tumor cell
    two 9's           - micro-metastasis 
    three or more 9's - macro-metastasis

    Each image contains between 3 and 10 cells at random, which may be overlapping.
    It consists of 5000 items of each category(total 20.000) for training and 500(2.000) of each for testing
    of size 256 x 256. 
    """

    def __init__(self,
                 root_dir,
                 train,
                 data_dir="MEDNIST",
                 mnist_transform=None,
                 transform=None,
                 total_train=20000,
                 total_test=2000,
                 n_partitions_test=1,
                 n_partitions_train=5):

        self.mnist_transform = mnist_transform
        self.root_dir = root_dir
        self.train = train
        self.total = total_train if self.train else total_test
        self.n_partitions_test = n_partitions_test
        self.n_partitions_train = n_partitions_train
        self.dir = os.path.join(root_dir, data_dir, "train" if train else "test")
        self.transform = transform

        self.__create_dataset_if_needed()

        self.__load_data()

    def __dataset_exists(self):
        # mkdir if not exists
        os.makedirs(self.dir, exist_ok=True)
        len_files = len(os.listdir(self.dir))
        if len_files > 0:
            print("Data existing, skipping creation.")
            return True
        else:
            print("Dataset missing. Creating...")
        return False

    def __combine_images(self, images, output_dim):
        """
        Combines the given images into a single image of output_dim size. Combinations are done randomly and 
        overlapping is possible. Images will always be within bounds completely.
        """
        np_images = np.array(images)
        input_dim = np_images.shape[-1]
        new_image = np.zeros(shape=(output_dim, output_dim), dtype=np.float32)
        for image in np_images:
            i, j = np.random.randint(0, output_dim - input_dim, size=2)
            new_image[i:i + input_dim, j:j + input_dim] = image
        return new_image

    def __get_cell_counts(self, items_per_class_count, class_index):
        # exclusive
        max_items = 11
        min_number_of_cells = 3
        # 0,1,2,3+ for no tumor cells, isolated tumor cells, 
        # micro-metastasis and macro-metastasis respectively
        num_tumor_cells = class_index if class_index != 3 else np.random.randint(3, max_items)

        num_healthy_cells = max_items - num_tumor_cells
        if num_healthy_cells + num_tumor_cells < min_number_of_cells:
            num_healthy_cells = min_number_of_cells - num_tumor_cells

        return (num_tumor_cells, num_healthy_cells)

    def __generate_for_class(self,
                             items,
                             items_per_class_count,
                             class_index,
                             uid,
                             all_tumor_cell_images,
                             all_healthy_cell_images):
        for _ in range(items_per_class_count):
            num_tumors, num_healthy = self.__get_cell_counts(items_per_class_count, class_index)

            healthy_idxs = np.random.randint(0, len(all_healthy_cell_images), num_healthy)
            tumor_idxs = np.random.randint(0, len(all_tumor_cell_images), num_tumors)

            healthy_cells = all_healthy_cell_images[healthy_idxs]
            tumor_cells = all_tumor_cell_images[tumor_idxs]
            cells = np.vstack((healthy_cells, tumor_cells))
            image = self.__combine_images(cells, 256)
            image = np.expand_dims(image, axis=0)
            self.data.append(image)
            self.source_images.append(tumor_cells.numpy())
            self.labels.append(class_index)
            uid += 1
        return uid

    def __create_dataset_if_needed(self):
        if self.__dataset_exists():
            return

        self.data = []
        self.labels = []
        self.source_images = []

        # in how many partitions to split dataset creation
        partitions_count = 10

        # number of classes in output (fixed)
        num_classes = 4

        mnist = torchvision.datasets.MNIST(root='./data',
                                           train=True,
                                           download=True,
                                           transform=self.mnist_transform)

        mnist_loader = iter(torch.utils.data.DataLoader(mnist,
                                                        batch_size=int(self.total / partitions_count),
                                                        shuffle=False,
                                                        num_workers=0))
        uid = 0
        batch, mnist_labels = mnist_loader.next()
        # 9's represent tumors
        all_tumor_cell_images = batch[mnist_labels == 9]
        # everything else except 6's healthy cells
        all_healthy_cell_images = batch[(mnist_labels != 9) & (mnist_labels != 6)]

        for _ in range(partitions_count):
            items_per_class_count = int(self.total / (num_classes * partitions_count))

            for class_index in range(num_classes):
                uid = self.__generate_for_class(class_index,
                                                items_per_class_count,
                                                class_index,
                                                uid,
                                                all_tumor_cell_images,
                                                all_healthy_cell_images)
        self.__store()
        print("Done.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, uid):
        if torch.is_tensor(uid):
            uid = uid.tolist()
        label = self.labels[uid]
        sample = self.data[uid]
        if self.transform:
            sample = self.transform(sample)

        return (sample, label)

    def __store(self):
        n_partitions = self.n_partitions_train if self.train else self.n_partitions_test

        assert (len(self.data) == len(self.labels))
        max_index = len(self.data)
        partition_size = max_index / n_partitions
        for i in range(n_partitions):
            start, end = (int(partition_size * i), int(partition_size * (i + 1)))
            partition = np.array(self.data[start:end])
            np.save(os.path.join(self.dir, "part_" + str(i)), partition)

        np.save(os.path.join(self.dir, "labels"), np.array(self.labels))

        #if not self.train:
        #    np.save(os.path.join(self.dir, "sources"), np.array(self.source_images))

    def __load_data(self):
        n_partitions = self.n_partitions_train if self.train else self.n_partitions_test
        data = []
        for i in range(n_partitions):
            data.append(np.load(os.path.join(self.dir, "part_" + str(i) + ".npy")))
        self.data = np.vstack(data)
        self.labels = np.load(os.path.join(self.dir, "labels.npy"))
        if not self.train:
            self.source_images = np.load(os.path.join(self.dir, "sources.npy"), allow_pickle=True)