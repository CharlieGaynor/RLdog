import torch.nn as nn


class StandardNN(nn.Module):
    """
    Only works for Discrete actions for now
    """

    def __init__(self, n_obs: int, n_actions: int) -> None:
        super(StandardNN, self).__init__()

        self.l1 = nn.Linear(n_obs, n_obs * 10)
        self.l2 = nn.Linear(n_obs * 10, n_obs * 5)
        self.l3 = nn.Linear(n_obs * 5, n_actions)
        self.activation = nn.ELU()

        self.sequential_network = nn.Sequential(
            self.l1, self.activation, self.l2, self.activation, self.l3
        )

    def forward(self, state):
        return self.sequential_network(state)
