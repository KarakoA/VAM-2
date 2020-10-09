import logging

import torch.nn as nn
import torch.nn.functional as F

from config.configs import Config
from network.glimpse_sensor import Retina


class GlimpseNetwork(nn.Module):
    """
    The glimpse network
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
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        logging.debug(phi.shape)

        # what
        # 3 conv layers
        h = F.relu(self.conv1(phi))
        h = F.relu(self.bn1(self.conv2(h)))
        h = F.relu(self.conv3(h))
        # flatten
        # keep batch dimension and determine other one automatically
        h = h.view(x.shape[0], -1)
        # fully connected layers
        h = F.relu(self.fc1(h))

        # where
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        l = F.relu(self.loc_fc1(l_t_prev))
        # combine what and where
        g = F.relu(h * l)

        return g