import logging

import torch
import torch.nn as nn

from network.action_network import ActionNetwork
from network.baseline_network import BaselineNetwork
from simplified.core_network_simple import CoreNetwork
from network.glimpse_network import GlimpseNetwork
from network.location_network import LocationNetwork


class RecurrentAttention(nn.Module):
    """A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References:
      [1]: Minh et. al., https://arxiv.org/abs/1406.6247
    """

    def __init__(self,config):
        """Constructor.

        Args:
          g: size of the square patches in the glimpses extracted by the retina.
          k: number of patches to extract per glimpse.
          s: scaling factor that controls the size of successive patches.
          c: number of channels in each image.
          h_g: hidden layer size of the fc layer for `phi`.
          h_l: hidden layer size of the fc layer for `l`.
          std: standard deviation of the Gaussian policy.
          hidden_size: hidden size of the rnn.
          num_classes: number of classes in the dataset.
          num_glimpses: number of glimpses to take per image,
            i.e. number of BPTT steps.
        """
        super().__init__()

        self.sensor = GlimpseNetwork(config)
        self.rnn = CoreNetwork(config.hidden_size, config.hidden_size)
        self.locator = LocationNetwork(config.hidden_size, 2, config.std)
        self.classifier = ActionNetwork(config.hidden_size, config.num_classes)
        self.baseliner = BaselineNetwork(config.hidden_size, 1)

    def forward(self, x, l_t_prev, last=False):
        """Run RAM for one timestep on a minibatch of images.

        Args:
            x: a 4D Tensor of shape (B, H, W, C). The minibatch
                of images.
            l_t_prev: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the previous
                timestep `t-1`.
            h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the previous timestep `t-1`.
            last: a bool indicating whether this is the last timestep.
                If True, the action network returns an output probability
                vector over the classes and the baseline `b_t` for the
                current timestep `t`. Else, the core network returns the
                hidden state vector for the next timestep `t+1` and the
                location vector for the next timestep `t+1`.

        Returns:
            h_t: a 2D tensor of shape (B, hidden_size). The hidden
                state vector for the current timestep `t`.
            mu: a 2D tensor of shape (B, 2). The mean that parametrizes
                the Gaussian policy.
            l_t: a 2D tensor of shape (B, 2). The location vector
                containing the glimpse coordinates [x, y] for the
                current timestep `t`.
            b_t: a vector of length (B,). The baseline for the
                current time step `t`.
            probabilities: a 2D tensor of shape (B, num_classes). The
                output log probability vector over the classes.
            mean_t: a vector of length (B,).
        """
        g_t = self.sensor(x, l_t_prev)
        h_t = self.rnn(g_t)
        mean_t, l_t = self.locator(h_t.detach())
        b_t = self.baseliner(h_t.detach()).squeeze()

        if last:
            probabilities = self.classifier(h_t)
            return h_t, l_t, b_t, probabilities, mean_t

        return h_t, l_t, b_t, mean_t

    def reset(self, batch_size, device):
        # h_t maintained by rnn itself
        self.rnn.reset(batch_size=batch_size, device=device)

        #l_t = torch.zeros(batch_size, 2).to(device)
        l_t = torch.FloatTensor(batch_size, 2).uniform_(-1, 1).to(device)
        logging.debug(f"DRAM reset, l_0: {l_t}")
        l_t.requires_grad = True

        return l_t