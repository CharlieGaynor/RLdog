#  type: ignore
import torch
from torch import nn
from agents.base_agent import agent_template


class eg_model(nn.Module, agent_template):
    """Epsilon Greedy model

    Args:
        n_obs (int): Dimensions of the state space (int for this project)
        n_actions (int): Number of possible actions
        lr (float, optional): Learning rate for the network. Defaults to 5e-4.
        epsilon_decay (float, optional): Multiplication factor for epsilon
    """

    def __init__(
        self,
        network: nn.Sequential,
        n_actions: int,
        lr: float = 1e-3,
        epsilon_decay: float = 0.01,
        alpha: float = 0.5,
        min_epsilon: float = 0.2,
    ):

        super().__init__()

        self.net = network
        self.n_actions = n_actions
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.alpha = alpha

    def forward(self, state):
        """Predicts Q values given state"""
        q_vals = self.net(state)
        return q_vals

    def update_epsilon(self, train):
        """update epsilon"""
        if train:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state: torch.FloatTensor, train: bool = False):
        """Sample actions with epsilon-greedy policy"""
        q_values = self.forward(state)

        ran_num = torch.rand(1)
        if ran_num < self.epsilon:
            return int(torch.randint(low=0, high=self.n_actions, size=(1,)))
        else:
            return int(torch.argmax(q_values))

    # def update_network(self):

    def update_network_old(
        self,
        states: torch.FloatTensor,
        actions: torch.LongTensor,
        rewards: torch.FloatTensor,
        next_states: torch.FloatTensor,
        done: torch.LongTensor,
        alpha: float = None,
        gamma: float = 0.95,
    ):
        if alpha is None:
            alpha = self.alpha

        self.opt.zero_grad()
        loss = self.compute_loss(
            states, actions, rewards, next_states, done, alpha, gamma
        )
        loss.backward()
        self.opt.step()

    def compute_loss(
        self,
        states: torch.FloatTensor,
        actions: torch.LongTensor,
        rewards: torch.FloatTensor,
        next_states: torch.FloatTensor,
        done: torch.LongTensor,
        alpha: float,
        gamma: float,
    ):
        """Computes loss function of the whole epoch

        states (torch.array[game_length]):  Observation for whole epoch.
        actions (torch.array[game_length]):  Actions over epoch
        rewards (torch.array[game_length]): Rewards over epoch
        next_states (torch.array[game_length -1]): States but shifted by 1
        done (torch.array[game_length]): 1 if the state is the finished one
        """
        states  # This may need to be deleted, depending on what we pass in :)
        q_values = self.forward(states)  # Shape [n_actions, game_length]
        # Shape [n_actions, game_length -1]
        next_q_values = self.forward(next_states)
        next_q_values_max = torch.max(
            next_q_values, dim=-1
        ).values  # Shape [game_length -1]

        q_values_updated = q_values.detach().clone()

        changed_q_values = q_values_updated[range(q_values.shape[0]), actions.flatten()]

        new_q_values = (
            changed_q_values * (1 - alpha)
            + alpha * rewards.flatten()
            + alpha * gamma * next_q_values_max
        )

        q_values_updated[range(q_values.shape[0]), actions.flatten()] = new_q_values

        # Shape [n_actions, game_length]
        loss = (q_values - q_values_updated) ** 2

        return torch.mean(loss)  # shape [1], loss over whole game
