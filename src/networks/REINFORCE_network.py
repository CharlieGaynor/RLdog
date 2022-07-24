import torch.nn as nn
from networks.base_network import baseNN


class reinforceNN(baseNN):
    """
    Only works for Discrete actions for now
    """

    def __init__(self, n_obs: int, n_actions: int, hidden_size: int) -> None:
        super(reinforceNN, self).__init__()

        self.l1 = nn.Linear(n_obs, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, n_actions)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        output = self.activation(self.l1(state))
        output = self.activation(self.l2(output))
        probabilities = self.softmax(self.l3(output))
        return probabilities
