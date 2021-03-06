import logging

import torch.nn as nn
import torch 
class CoreNetwork(nn.Module):
    """The core network.
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.num_layers = 1
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.hidden_state = None
        self.cell_state = None

        # batch_first = true -> (B x SEQ (in this case 1) x Features)
        # and SEQ only 1 element
        self.stacked_lstm = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=self.num_layers,
                                    batch_first=True)
        logging.info(self)

    def reset(self, batch_size, device):
        self.hidden_state = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            dtype=torch.float,
            device=device,
            requires_grad=True)

        self.cell_state = torch.zeros(
            (self.num_layers, batch_size, self.hidden_size),
            dtype=torch.float,
            device=device,
            requires_grad=True)

    def forward(self, g_t):
        # need to add seq dimension
        g_t = g_t.unsqueeze(1)
        logging.debug(f"Input(Seq): {g_t.shape}")
        (output, (self.hidden_state, self.cell_state)) = self.stacked_lstm.forward(g_t,
                                                       (self.hidden_state, self.cell_state))
        # remove seq dimension
        return output.squeeze()