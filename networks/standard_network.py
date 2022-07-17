import torch.nn as nn
from networks.base_network import baseNN


class standardNN(baseNN):
    """
    Only works for Discrete actions for now
    """

    def __init__(self, n_obs: int, n_actions: int) -> None:
        super(standardNN, self).__init__()

        self.l1 = nn.Linear(n_obs, n_obs)
        self.l2 = nn.Linear(n_obs, n_obs)
        self.l3 = nn.Linear(n_obs, n_actions)
        self.activation = nn.ReLU()

    def forward(self, state):
        output = self.activation(self.l1(state))
        output = self.activation(self.l2(output))
        return self.l3(output)
