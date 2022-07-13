import torch
from torch import nn
import numpy as np
from icecream import ic
import pickle
import copy
from base_agent import agent_template


class ac_model(nn.Module, agent_template):
    """Epsilon Greedy model"""

    def __init__(
        self,
        network: nn.Sequential,
        n_actions: int,
        lr: float = 1e-3,
    ):
        """
        Args:
            n_obs (int): Dimensions of the state space (int for this project)
            n_actions (int): Number of possible actions
            lr (float, optional): Learning rate for the network. Defaults to 5e-4.
        """
        super().__init__()

        self.net = network
        self.n_actions = n_actions
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epsilon = 1
        self.lr = lr

    def forward(self, state) -> torch.FloatTensor:
        """Predicts Q values given state

        Args:
            state (np.array): Observation for given step

        Returns:
            q_values
        """
        probabilites = self.net(state)
        return probabilites

    def get_action(self, state: torch.FloatTensor, train: bool = False):
        """Sample actions with epsilon-greedy policy

        Args:
            state (np.array): Observation for a given step
            epsilon (float, optional): Exploration probability. Defaults to 0.

        Returns:
            int: Action to take (card to play)
        """
        probabilites = self.forward(state)

        numpy_probabilities: np.array = probabilites.data.flatten().numpy()

        if not train:
            trimmed_probs = np.where(numpy_probabilities < 0.05, 0, numpy_probabilities)
            numpy_probabilities = trimmed_probs / np.sum(trimmed_probs)

        action = np.random.choice(len(numpy_probabilities), p=numpy_probabilities)
        return action

    def update_network(
        self,
        states: torch.FloatTensor,
        actions: torch.LongTensor,
        rewards: torch.FloatTensor,
        gamma: float = 0.99,
    ):

        self.opt.zero_grad()
        loss = self.compute_loss(states, actions, rewards, gamma)
        loss.backward()
        self.opt.step()

    def compute_loss(
        self,
        states: torch.FloatTensor,  # [H, n_obs]
        actions: torch.LongTensor,  # [H]
        rewards: torch.FloatTensor,  # [H]
        gamma: float = 0.95,
    ):
        """Computes loss function of the whole epoch

        states (torch.array[game_length]):  Observation for whole epoch.
        actions (torch.array[game_length]):  Actions over epoch
        rewards (torch.array[game_length]): Rewards over epoch
        next_states (torch.array[game_length -1]): States but shifted by 1
        done (torch.array[game_length]): 1 if the state is the finished one
        """
        probabilities = self.forward(states)
        relevant_probabilities = probabilities.gather(dim=1, index=actions.long().view(-1, 1)).squeeze()
        g_t = self.calculate_g_t(rewards, gamma)

        # if not isinstance(g_t, torch.Tensor):
        #     print('here')
        #     g_t = self.calculate_g_t(rewards, gamma, verbose = True)

        loss = - torch.sum(torch.log(relevant_probabilities) * g_t)

        if torch.isnan(loss):
            print('probabilities:', probabilities)
            print('relevant_probs', relevant_probabilities)
            print('g_t', g_t)
            print(loss)
        return loss  # shape [1], loss over whole game

    @staticmethod
    def calculate_g_t(rewards: torch.FloatTensor, gamma: float, verbose: bool = False) -> torch.FloatTensor:

        g_t = []

        for i in range(len(rewards)):
            total = 0
            for j in range(i, len(rewards)):
                total += rewards[j] * gamma ** (j - i)

            g_t.append(total)

        max_g = max(g_t)

        # if verbose:
        #     ic(g_t)
        #     ic(total)
        #     ic(rewards)
        #     ic(gamma)
        #     ic(max_g)

        return torch.FloatTensor(g_t) / max_g if max_g != torch.tensor(0.) else torch.FloatTensor(g_t)
