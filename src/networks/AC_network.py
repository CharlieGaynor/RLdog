import torch.nn as nn
import torch
from typing import Tuple
from networks.base_network import baseNN


class acNN(baseNN):
    """
    AC network where both the critic and actor share a network. Not used right now
    """

    def __init__(self, n_obs: int, n_actions: int) -> None:
        super(acNN, self).__init__()

        self.l1 = nn.Linear(n_obs, 128)
        self.l2 = nn.Linear(128, 128)
        self.actor_layer = nn.Linear(128, n_actions)
        self.critic_layer = nn.Linear(128, 1)
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        output = self.activation(self.l1(state))
        output = self.activation(self.l2(output))
        actor_probs = self.softmax(self.actor_layer(output))
        critic_value = self.critic_layer(output)
        return actor_probs, critic_value
