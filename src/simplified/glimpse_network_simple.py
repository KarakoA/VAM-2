import logging

import torch.nn as nn
import torch.nn.functional as F

from src.config.configs import Config
from src.network.glimpse_sensor import Retina


class GlimpseNetwork(nn.Module):
    """The glimpse network.


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

        self.fc1 = nn.Linear(in_features=self.retina.num_patches * self.retina.patch_size * self.retina.patch_size, out_features=conf.glimpse_hidden)
        self.fc2 = nn.Linear(in_features=conf.glimpse_hidden, out_features=D_out)
        self.bn2 = nn.BatchNorm1d(num_features=D_out, track_running_stats=True)

        # where
        # in_features = 2, loc is a tuple of (x,y)
        self.loc_fc1 = nn.Linear(in_features=2, out_features=conf.loc_hidden)
        self.loc_fc2 = nn.Linear(in_features=conf.loc_hidden, out_features=D_out)

        logging.info(self)

    def forward(self, x, l_t_prev):
        logging.debug("\n\nGlimpseNetwork shapes")
        logging.debug("#### What ####")
        # generate glimpse phi from image x
        phi = self.retina.foveate(x, l_t_prev)
        logging.debug(phi.shape)

        # flatten
        # keep batch dimension and determine other one automatically
        h = phi.view(x.shape[0], -1)
        logging.debug(f"Flatten:    {h.shape}")

        # fully connected layers
        h = self.fc1(h)
        logging.debug(f"Fc1:        {h.shape}")
        h = self.fc2(h)
        logging.debug(f"Fc2:        {h.shape}")
        h = self.bn2(h)
        logging.debug(f"Bn1:        {h.shape}")
        h = F.relu(h)
        logging.debug(f"Bn1 ReLu:   {h.shape}")

        # where
        logging.debug("#### Where ####")
        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
        logging.debug(f"Input:         {l_t_prev.shape}")

        l = self.loc_fc1(l_t_prev)
        logging.debug(f"Fc1(loc):      {l.shape}")
        l = F.relu(l)
        logging.debug(f"Fc1(loc) ReLu: {l.shape}")
        l = self.loc_fc2(l)
        logging.debug(f"Fc2(loc):      {l.shape}")
        logging.debug("#### Combined ####")
        # combine what and where
        g = F.relu(h * l)

        logging.debug(f"relu(h * l):   {g.shape}\n\n\n")
        return g