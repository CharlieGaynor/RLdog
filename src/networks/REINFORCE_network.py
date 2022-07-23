from black import out
import torch.nn as nn
from networks.base_network import baseNN


class reinforceNN(baseNN):
    """
    Only works for Discrete actions for now
    """

    def __init__(self, n_obs: int, n_actions: int) -> None:
        super(reinforceNN, self).__init__()

        self.l1 = nn.Linear(n_obs, n_obs*5)
        self.l2 = nn.Linear(n_obs*5, n_obs*5)
        self.l3 = nn.Linear(n_obs*5, n_actions)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        output = self.activation(self.l1(state))
        output = self.activation(self.l2(output))
        probabilities = self.softmax(self.l3(output))
        return probabilities
