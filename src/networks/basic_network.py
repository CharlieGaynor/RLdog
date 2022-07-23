import torch.nn as nn
from networks.base_network import baseNN


class basicNN(baseNN):
    """
    Only works for Discrete actions for now
    """

    def __init__(self, n_obs: int, n_actions: int) -> None:
        super(basicNN, self).__init__()

        self.l1 = nn.Linear(n_obs, n_actions)

    def forward(self, state):
        return self.l1(state)
