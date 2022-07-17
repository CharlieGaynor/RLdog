import torch.nn as nn
from networks.base_network import baseNN
import torch


class testNN(baseNN):
    """
    Only works for Discrete actions for now
    """

    def __init__(self, n_obs: int, n_actions: int) -> None:
        super(testNN, self).__init__()

        self.l1 = nn.Linear(n_obs, n_actions)
        self.l1.bias.data.fill_(0)

    def forward(self, state):
        return self.l1(state)
