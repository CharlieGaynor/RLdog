from typing import Any, List
import torch
from torch import nn
import numpy as np
import pickle
import copy
from agents.base_agent import agent_template


## at the minute using epislon greedy - could generalise this out into a seperate class
## priority is having mini batches

class DQN(nn.Module, agent_template):
    """Epsilon Greedy model

    Args:
        n_obs (int): Dimensions of the state space (int for this project)
        n_actions (int): Number of possible actions
        lr (float, optional): Learning rate for the network. Defaults to 5e-4.
        epsilon_decay (float, optional): Multiplication factor for epsilon
    """

    def __init__(
        self,
        network: nn
        n_actions: int,
        lr: float = 1e-3,
        epsilon_decay: float = 0.01,
        alpha: float = 0.5,
        min_epsilon: float = 0.2,
    ):

        super().__init__()

        self.network = network
        self.n_actions = n_actions
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.env = env
        self.alpha = alpha
        self.mini_batch_size = mini_batch_size
        self.transitions = deque([], maxlen=64)

    def update_epsilon(self, train):
        pass

    def get_action(self, state: torch.FloatTensor, train: bool = False):
        """Sample actions with epsilon-greedy policy"""
        q_values = self.network(state)

        if len(q_values) == 1:
            return q_values

        ran_num = torch.rand(1)
        if ran_num < self.epsilon:
            return int(torch.randint(low=0, high=self.n_actions, size=(1,)))
        else:
            return int(torch.argmax(q_values))

    def _play_game(self):
        """Plays out one game"""
        next_obs = self.env.reset()
        done = False
        while not done:
            obs = torch.tensor(next_obs, dtype=torch.float32)
            action = self.get_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            self.transitions.appendleft([obs, action, reward, next_obs, done])

    def play_games(self, games_to_play):

        while games_to_play > 1:
            self._play_game()
            if self.network_needs_updating():
                self.update_network()
            games_to_play -= 1

        
    def network_needs_updating(self):
        """For standard DQN, network needs updated if self.transitions contains more than
        self.mini_batch_size items"""
        return len(self.transitions) >= self.mini_batch_size

    def sample_experiences(self) -> torch.Tensor[List[Any]]:
        """
        Returns list of experiences with dimensions [mini_batch_size]
        """
        experiences = []
        for _ in range(self.mini_batch_size):
            experiences.append(self.transitions.pop())
        return torch.tensor(experiences)

    def update_network(self):

        experiences = self.sample_experiences()  # shape [mini_batch_size]
        print(experiences)

        loss = self.compute_loss()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def compute_loss(self, obs, actions, rewards, next_obs, done):

        current_q_vals = self.calculate_current_q_values(obs)
        target_q_vals = self.calculate_target_q_values(current_q_vals, next_obs)      
        
        loss = torch.mean((target_q_vals - current_q_vals) ** 2)
        return loss

    def calculate_current_q_values(self, obs):
        q_vals = self.network(obs)
        return q_vals

    def calculate_target_q_values0(self, current_q_vals, next_obs):

        next_q_vals = self.forward(next_obs)
        target_q_vals_max = torch.max(
            next_q_vals, dim=-1).values

        q_values_clone = current_q_vals.detach().clone()

        # What were the Q values for the actions we took?
        relevant_q_values = q_values_clone[range(
            current_q_vals.shape[0]), actions.flatten()]

        # What should the Q values be updated to for the actions we took?
        target_q_vals = relevant_q_values * \
            (1 - self.alpha) + self.alpha * rewards.flatten() + \
            self.alpha * self.gamma * target_q_vals_max
        return target_q_vals