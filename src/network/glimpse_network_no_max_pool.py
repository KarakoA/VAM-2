import logging

import torch.nn as nn
import torch.nn.functional as F

from config.configs import Config
from network.glimpse_sensor import Retina


class GlimpseNetwork(nn.Module):
    """The glimpse network.

a    TODO docs

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

    def __init__(self, conf: Config):
        super().__init__()

        self.retina = Retina(conf)

        D_out = conf.glimpse_hidden + conf.loc_hidden

        # what

        # padding of 1, to ensure same dimensions
        self.conv1 = nn.Conv2d(in_channels=self.retina.num_patches, out_channels=16, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv2.out_channels, track_running_stats=True)

        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=16, kernel_size=3, padding=1)

        D_in = self.conv3.out_channels * conf.patch_size * conf.patch_size

        self.fc1 = nn.Linear(in_features=D_in, out_features=D_out)

        # where
        # in_features = 2, loc is a tuple of (x,y)
        self.loc_fc1 = nn.Linear(in_features=2, out_features=D_out)

        logging.info(self)

    def forward(self, x, l_t_prev):
        logging.debug("\n\nGlimpseNetwork shapes")
        logging.debug("#### What ####")
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        logging.debug(phi.shape)

        # what
        # 3 conv layers
        h = self.conv1(phi)
        logging.debug(f"Conv1:      {h.shape}")
        h = F.relu(h)
        logging.debug(f"Conv1 ReLu: {h.shape}")
        h = F.relu(self.bn1(self.conv2(h)))
        logging.debug(f"Conv2:        {h.shape}")
        logging.debug(f"Bn1 ReLu:   {h.shape}")
        h = F.relu(self.conv3(h))
        logging.debug(f"Conv3:      {h.shape}")
        # flatten
        # keep batch dimension and determine other one automatically
        h = h.view(x.shape[0], -1)
        logging.debug(f"Flatten:    {h.shape}")

        # fully connected layers
        h = F.relu(self.fc1(h))
        logging.debug(f"Fc1:        {h.shape}")

        # where
        logging.debug("#### Where ####")
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        logging.debug(f"Input:         {l_t_prev.shape}")

        l = F.relu(self.loc_fc1(l_t_prev))
        logging.debug(f"Fc1(loc):      {l.shape}")
        logging.debug("#### Combined ####")
        # combine what and where
        g = F.relu(h * l)

        logging.debug(f"relu(h * l):   {g.shape}\n\n")
        return g
