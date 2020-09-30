import os
import pickle
import matplotlib.patches as patches

from config.configs import Config


def denormalize(T, coords):
    return 0.5 * ((coords + 1.0) * T)


def bounding_box(x, y, size, color="w"):
    x = int(x - (size / 2))
    y = int(y - (size / 2))
    rect = patches.Rectangle(
        (x, y), size, size, linewidth=1, edgecolor=color, fill=False
    )
    return rect

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def images_and_locations(epoch, plot_dir):
    images = pickle.load(open(plot_dir + "g_{}.p".format(epoch), "rb"))
    locations = pickle.load(open(plot_dir + "l_{}.p".format(epoch), "rb"))
    images = np.concatenate(images)
    return (images,locations)

def run(epoch1,epoch2):

    config = Config()
    plot_dir = "./plots/" + config.model_name + "/"

    # read in pickle files
    (images1,locations1)  = images_and_locations(epoch1,plot_dir)

    (images2, locations2) = images_and_locations(epoch2,plot_dir)

    assert (images1 == images2).all()
    size = int(plot_dir.split("_")[2].split("x")[0])
    num_anims = len(locations1)

    num_cols = images1.shape[0]
    img_shape = images1.shape[1]

    # denormalize coordinates
    coords1 = [denormalize(img_shape, l) for l in locations1]
    coords2 = [denormalize(img_shape, l) for l in locations2]
    fig, axs = plt.subplots(nrows=1, ncols=num_cols)
    fig.set_dpi(400)

    labels =[2, 1, 0, 1, 2, 3, 2, 1, 3, 1]
    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(images1[j], cmap="Greys_r")
        ax.set_title(labels[j])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    #cols = ["r","g","b","y","m","c"]
    k = list(zip(coords1, coords2))
    def updateData2(i):
        (coords1, coords2) = k[i]
        # j - img counter
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            for p in ax.patches:
                p.remove()
            co1 = coords1[j]
            co2 = coords2[j]
            rect1 = bounding_box(co1[0], co1[1], size, "r")

            rect2 = bounding_box(co2[0], co2[1], size, "b")

            ax.add_patch(rect1)
            ax.add_patch(rect2)
    # animate
    anim = animation.FuncAnimation( fig, updateData2, frames=range(len(coords1)), interval=750, repeat=False )
    # save as mp4
    name = plot_dir + "epoch_{}=={}.mp4".format(epoch1,epoch2)
    anim.save(name, extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"])
    print(name)
    os.system("open "+name)

if __name__ == '__main__':
    epoch1 = 42
    epoch2 = 43
    run(epoch1,epoch2)