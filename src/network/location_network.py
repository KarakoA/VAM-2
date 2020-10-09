import logging

import torch.nn as nn
import torch


class LocationNetwork(nn.Module):
    """
    The location network.
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        self.fc = nn.Linear(input_size, output_size)
        logging.info(self)

    def forward(self, h_t):
        # compute mean
        mean = torch.tanh(self.fc(h_t.detach()))

        if self.training:
            l_t = torch.distributions.Normal(mean, self.std).rsample().detach()
        # eval, not stochastic
        else:
            l_t = mean

        l_t = torch.clamp(l_t, -1, 1)

        return mean, l_t
