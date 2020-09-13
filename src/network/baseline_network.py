import logging

import torch.nn as nn


class BaselineNetwork(nn.Module):
    """The baseline network.

    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.

    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        h_t: the hidden state vector of the core network
            for the current time step `t`.

    Returns:
        b_t: a 2D vector of shape (B, 1). The baseline
            for the current time step `t`.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)
        logging.info(self)

    def forward(self, h_t):
        logging.debug("\n\nBaselineNetwork")
        logging.debug(f"Input: {h_t.shape}")
        b_t = self.fc(h_t.detach())
        logging.debug(f"Fc1:   {b_t.shape}\n\n")
        return b_t