import torch
import matplotlib.pyplot as plt
import numpy as np
from src.config.configs import Config
from src.datasets.closed_squares import ClosedSquaresDataset
from src.datasets.datasets import DatasetType
from src.datasets.locator import DatasetLocator
from src.network.glimpse_sensor import Retina

import scipy.misc

def show_patches(images, labels, retina, locs, suptitle = ""):
    extracted_patches = retina.foveate(images, locs)

    extracted_patches = extracted_patches.reshape(
        (len(images), retina.num_patches, retina.patch_size, retina.patch_size))
    fig, axes = plt.subplots(len(images), retina.num_patches + 1, figsize=(25, 25))

    for patches, image, axis,label_str in zip(extracted_patches, images, axes,labels):
        for i in range(retina.num_patches + 1):
            im = image[0] if i == 0 else patches[i - 1]
            scale = retina.scale ** (i - 1)
            label = label_str if i == 0 else "size: " + str(
                retina.patch_size) + " x " + str(retina.patch_size) + " scale: " + str(scale)
            axis[i].imshow(im.numpy(), cmap=plt.cm.gray)
            axis[i].set_title(label, fontsize=20)
    fig.suptitle(suptitle, fontsize=20)
    plt.show()

def run():
    conf = Config()
    glimpse_sensor = Retina(conf)
    test = ClosedSquaresDataset(train=False)
    locator = DatasetLocator(conf)
    # one of each class
    images, labels = test[0:400:100]
    images = torch.cat(images).unsqueeze(1)

    locs = torch.tensor([[0, 0],
                     [-1, -1],
                    [1, 1],
                    [0,1],
                    [0,-1],
                    [0.5,0.5]]
                    )
    labels = ["Center", "Top Left", "Bottom Right", "Bottom Center", "Top Center", "Random"]
    images = scipy.misc.face()[:, :, 0]
    images = resize(images, (64, 64))
    n = len(labels)

    images = np.repeat(images, n).reshape((images.shape[0], images.shape[1], n)).transpose(2, 0, 1)
    images = torch.tensor(images).unsqueeze(1)
    show_patches(images,labels,glimpse_sensor,locs)

from skimage.transform import resize

# and other with patches from mine
if __name__ == '__main__':
    run()
