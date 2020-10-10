import logging
import math

import torch.nn as nn
import torch.nn.functional as F

from config.configs import Config
from network.glimpse_sensor import Retina


class GlimpseNetwork(nn.Module):
    """
    The glimpse network.
    """

    def __init__(self, conf:Config):
        super().__init__()

        self.retina = Retina(conf)
        
        D_out = conf.glimpse_hidden + conf.loc_hidden

        # what

        # padding of 1, to ensure same dimensions
        self.conv1 = nn.Conv2d(in_channels=self.retina.num_patches, out_channels=16, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv2.out_channels, track_running_stats=True)

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=32, kernel_size=3, padding=1)

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        # W * H of previous layer * depth
        # W* H altered by max pooling
        D_in = self.conv3.out_channels# * reduced_dim * reduced_dim

        self.fc1 = nn.Linear(in_features=D_in, out_features=conf.glimpse_hidden)
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

        # what
        # 3 conv layers
        h = self.conv1(phi)
        logging.debug(f"Conv1:      {h.shape}")
        h = F.relu(h)
        logging.debug(f"Conv1 ReLu: {h.shape}")
        h = self.conv2(h)
        logging.debug(f"Conv2:      {h.shape}")
        h = self.bn1(h)
        logging.debug(f"Bn1:        {h.shape}")
        h = F.relu(h)
        logging.debug(f"Bn1 ReLu:   {h.shape}")
        h = self.max_pool1(h)
        logging.debug(f"MaxPool1:   {h.shape}")
        h = F.relu(self.conv3(h))
        logging.debug(f"Conv3:      {h.shape}")
        h = self.max_pool2(h)
        logging.debug(f"MaxPool2:   {h.shape}")

        # flatten
        # keep batch dimension and determine other one automatically
        h = h.view(x.shape[0], -1)
        logging.debug(f"Flatten:    {h.shape}")

        # fully connected layers
        h = F.relu(self.fc1(h))
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
        #l = F.relu(l)
        logging.debug(f"Fc1(loc) ReLu: {l.shape}")
        l = self.loc_fc2(l)
        #l = F.relu(l)
        logging.debug(f"Fc2(loc):      {l.shape}")
        logging.debug("#### Combined ####")
        # combine what and where
        g = F.relu(h * l)

        logging.debug(f"relu(h * l):   {g.shape}\n\n")
        return g