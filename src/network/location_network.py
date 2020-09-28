import logging

import torch.nn as nn
import torch
from torch.distributions import Normal

import torch.nn.functional as F

class LocationNetwork(nn.Module):
    """The location network.

    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.

    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.

    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.

    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, output_size, std):
        super().__init__()

        self.std = std

        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, output_size)
        #self.fc_lt = nn.Linear(hid_size, output_size)
        logging.info(self)

    def forward(self, h_t):
        logging.debug("\n\nLocationNetwork")
        logging.debug(f"Input:     {h_t.shape}")
        # compute mean
        mean = torch.tanh(self.fc(h_t.detach()))
        logging.debug(f"fc2+tanh:  {mean.shape}")

        if self.training:
            l_t = torch.distributions.Normal(mean, self.std).rsample().detach()
        #eval, not stochastic
        else:
            l_t = mean

        l_t = torch.zeros(25, 2).detach()
        #if torch.any(l_t < -1):
            #print("MEAN")
            #print(mean)
            #print("L_T")
            #print(l_t)
        ##if torch.any(l_t > 1):
        #    print("MEAN")
        #    print(mean)
        # bound between [-1, 1]
        l_t = torch.clamp(l_t, -1, 1)

        return mean, l_t