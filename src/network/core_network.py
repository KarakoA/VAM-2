import torch.nn as nn
import torch 
class CoreNetwork(nn.Module):
    """The core network.
    TODO
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.num_layers = 2
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
        # output == top layer of h_t. So for 2 layers `o == h_t[1]` yield all true
        (output, (self.cell_state, self.hidden_state)) = self.stacked_lstm.forward(g_t,
                                                                                   (self.cell_state, self.hidden_state))
        # remove seq dimension
        return output.squeeze()