import torch.nn as nn
import torch


class StandardNN(nn.Module):
    """
    Only works for Discrete actions for now 
    """
    def __init__(self, n_obs, n_actions) -> None:
        super(StandardNN, self).__init__()

        self.l1 = nn.Linear(n_obs, n_obs * 10)
        self.l2 = nn.Linear(n_obs*10, n_actions)


    def forward(self, state):
        pass1 = nn.ELU(self.l1(state))
        return self.l2(pass1)