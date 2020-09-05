from torch import Tensor
import matplotlib.pyplot as plt


def show_image(image, title=""):
    image = __format_image(image)
    plt.imshow(image, label = title, cmap=plt.cm.gray)
    plt.show()

def __format_image(image):
    if image is Tensor:
        image = image.numpy()
    shape = image.shape
    if len(shape) == 3:
        if (shape[0] == 0 or shape[0] == 1):
            image = image.reshape(image.shape[1], image.shape[2])
        else:
            raise Exception("multiple images passed to format image")
    return image

def show_images(images, labels):
    images = [__format_image(img) for img in images]
    fig, axes = plt.subplots(1,len(images), figsize=(25,25))
    for ax,image,label in zip(axes,images,labels):
        ax.imshow(image, cmap=plt.cm.gray)
        ax.set_title(label, fontsize=20)
    fig.show()
