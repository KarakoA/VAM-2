import os
import pickle
import argparse
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

def run():
    config = Config()
    plot_dir = "./plots/" + config.model_name + "/"
    epoch = 163
    # read in pickle files
    images = pickle.load(open(plot_dir + "g_{}.p".format(epoch), "rb"))
    locations = pickle.load(open(plot_dir + "l_{}.p".format(epoch), "rb"))
    images = np.concatenate(images)

    size = int(plot_dir.split("_")[2].split("x")[0])
    num_anims = len(locations)

    num_cols = images.shape[0]
    img_shape = images.shape[1]
    # denormalize coordinates
    coords = [denormalize(img_shape, l) for l in locations]
    fig, axs = plt.subplots(nrows=1, ncols=num_cols)
    fig.set_dpi(400)


    labels =[2, 1, 0, 1, 2, 3, 2, 1, 3, 1]

    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(images[j], cmap="Greys_r")
        print(j)
        ax.set_title(labels[j])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    def updateData(i):
        color = "r"

        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = bounding_box(c[0], c[1], size, color)

            ax.add_patch(rect)

                # animate
    anim = animation.FuncAnimation( fig, updateData, frames=num_anims, interval=500, repeat=True )
    #from IPython.display import HTML HTML(anim.to_html5_video())
    #anim.to_html5_video()
    # save as mp4
    name = plot_dir + "epoch_{}.mp4".format(epoch)
    anim.save(name, extra_args=["-vcodec", "h264", "-pix_fmt", "yuv420p"])
    print(name)
    os.system("open "+name)

if __name__ == '__main__':
    run()