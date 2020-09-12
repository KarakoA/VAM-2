import math

import torch.nn as nn
import torch.nn.functional as F

from src.config.configs import Config
from src.network.glimpse_sensor import Retina


class GlimpseNetwork(nn.Module):
    """The glimpse network.

    TODO

    Args:
        conf.glimpse_hidden: hidden layer size of the fc layer for `phi`.
        conf.loc_hidden: hidden layer size of the fc layer for `l`.
        g: size of the square patches in the glimpses extracted
        by the retina.
        k: number of patches to extract per glimpse.
        s: scaling factor that controls the size of successive patches.
        c: number of channels in each image.
        x: a 4D Tensor of shape (B, H, W, C). The minibatch
            of images.
        l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
            coordinates [x, y] for the previous timestep `t-1`.

    Returns:
        g_t: a 2D tensor of shape (B, hidden_size).
            The glimpse representation returned by
            the glimpse network for the current
            timestep `t`.
    """

    def __init__(self, conf:Config):
        super().__init__()

        self.retina = Retina(conf)
        
        D_out = conf.glimpse_hidden + conf.loc_hidden

        # what

        # padding of 1, to ensure same dimensions
        self.conv1 = nn.Conv2d(in_channels=self.retina.patch_size, out_channels=16, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv2.out_channels, track_running_stats=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=3, padding=1)

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        # W * H of previous layer * depth
        # W* H altered by max pooling
        reduced_dim = math.floor((1 + math.floor((1 + 12 + 1) / 3) + 1) / 3)
        D_in = self.conv3.out_channels * reduced_dim * reduced_dim

        self.fc1 = nn.Linear(in_features=D_in, out_features=conf.glimpse_hidden)
        self.fc2 = nn.Linear(in_features=conf.glimpse_hidden, out_features=D_out)
        self.bn2 = nn.BatchNorm1d(num_features=D_out, track_running_stats=True)

        # where
        # in_features = 2, loc is a tuple of (x,y)
        self.loc_fc1 = nn.Linear(in_features=2, out_features=conf.loc_hidden)
        self.loc_fc2 = nn.Linear(in_features=conf.loc_hidden, out_features=D_out)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        # what
        # 3 conv layers
        h = self.conv1(phi)
        h = F.relu(h)
        h = self.conv2(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = self.max_pool1(h)
        h = F.relu(self.conv3(h))
        h = self.max_pool2(h)

        # flatten
        # keep batch dimension and determine other one automatically
        h = h.view(x.shape[0], -1)

        # fully connected layers
        h = self.fc1(h)
        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)

        # where
        l = self.loc_fc1(l_t_prev)

        l = F.relu(l)
        l = self.loc_fc2(l)

        # combine what and where
        g = F.relu(h * l)
        return g