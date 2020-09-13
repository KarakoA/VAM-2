import torch.nn as nn
import torch
import torch.nn.functional as F

class CoreNetwork(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, g_t):
        h1 = self.i2h(g_t)
        h2 = self.h2h(self.hidden_state)
        h_t = F.relu(h1 + h2)
        return h_t

    def reset(self, batch_size, device):
        self.hidden_state = torch.zeros(
            (batch_size, self.hidden_size),
            dtype=torch.float,
            device=device,
            requires_grad=True)