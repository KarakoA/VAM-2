import logging

import torch.nn as nn
import torch.nn.functional as F

from config.configs import Config
from network.glimpse_sensor import Retina


class GlimpseNetwork(nn.Module):
    """
    The glimpse network.
    """

    def __init__(self, conf: Config):
        super().__init__()

        self.retina = Retina(conf)

        D_out = conf.glimpse_hidden + conf.loc_hidden

        self.fc1 = nn.Linear(in_features=self.retina.num_patches * self.retina.patch_size * self.retina.patch_size, out_features=conf.glimpse_hidden)
        self.fc2 = nn.Linear(in_features=conf.glimpse_hidden, out_features=D_out)

        # where
        # in_features = 2, loc is a tuple of (x,y)
        self.loc_fc1 = nn.Linear(in_features=2, out_features=conf.loc_hidden)
        self.loc_fc2 = nn.Linear(in_features=conf.loc_hidden, out_features=D_out)

        logging.info(self)

    def forward(self, x, l_t_prev):
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        logging.debug(phi.shape)

        # flatten
        # keep batch dimension and determine other one automatically
        h = phi.view(x.shape[0], -1)
        # fully connected layers
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        # where
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        l = F.relu(self.loc_fc1(l_t_prev))
        l = self.loc_fc2(l)
        # combine what and where
        g = F.relu(h * l)

        return g