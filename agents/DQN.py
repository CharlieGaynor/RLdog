from typing import List, Any
import torch
from torch import nn
import numpy as np
from collections import deque
import gym
import constants as const
from collections import Counter
from networks import base_network as bn

# at the minute using epislon greedy - could generalise this out into a seperate class
# priority is having mini batches


class DQN(nn.Module):
    """Epsilon Greedy model

    Args:
        n_obs (int): Dimensions of the state space (int for this project)
        n_actions (int): Number of possible actions
        lr (float, optional): Learning rate for the network. Defaults to 5e-4.
        epsilon_decay (float, optional): Multiplication factor for epsilon
    """

    def __init__(
        self,
        network: bn.StandardNN,
        n_actions: int,
        env: gym.Env,
        max_games: int = 10000,
        lr: float = 1e-1,
        epsilon_decay: float = 0.01,
        alpha: float = 0.5,
        min_epsilon: float = 0.2,
        mini_batch_size=1,
        buffer_size=128,
    ):

        super().__init__()

        self.network = network
        self.n_actions = n_actions
        self.opt = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.epsilon = 1
        self.max_games = max_games
        self.epsilon_decay = min_epsilon ** (1 / max_games)
        self.min_epsilon = min_epsilon
        self.env = env
        self.alpha = alpha
        self.gamma = 0.99
        self.reward_averages: list[list[float]] = []
        self.action_counts = {i: 0 for i in range(self.n_actions)}
        self.mini_batch_size = mini_batch_size
        self.transitions: deque[List[Any]] = deque([], maxlen=buffer_size)

    def update_epsilon(self, train):
        pass

    def get_action(self, state: torch.Tensor, train: bool = False):
        """Sample actions with epsilon-greedy policy"""
        q_values = self.network(state)

        if len(q_values) == 1:
            return q_values

        ran_num = torch.rand(1)
        if ran_num < self.epsilon:
            return int(torch.randint(low=0, high=self.n_actions, size=(1,)))
        else:
            return int(torch.argmax(q_values))

    def _play_game(self) -> None:
        """Plays out one game"""
        next_obs: np.ndarray = self.env.reset()  # type: ignore
        done = False
        rewards = []
        while not done:
            obs = next_obs
            action = self.get_action(torch.tensor(obs))
            next_obs, reward, done, _ = self.env.step(action)  # type: ignore
            rewards.append(reward)
            self.transitions.appendleft(
                [obs.tolist(), [action], [reward], next_obs.tolist(), [done]]
            )

        self.epsilon *= self.epsilon_decay
        self.reward_averages.append([0.0, sum(rewards)])

    def _play_games(self, games_to_play: int) -> None:

        while games_to_play > 1:
            self._play_game()
            if self.network_needs_updating():
                self.update_network()
            games_to_play -= 1

    def network_needs_updating(self) -> bool:
        """For standard DQN, network needs updated if self.transitions contains more than
        self.mini_batch_size items"""
        return len(self.transitions) >= self.mini_batch_size

    def sample_experiences(self) -> np.ndarray:
        """
        Returns list of experiences with dimensions [mini_batch_size]
        """
        experiences = []
        for _ in range(self.mini_batch_size):
            element = self.transitions.pop()
            experiences.append(element)

        return np.array(experiences, dtype=object)

    def update_network(self):

        experiences = self.sample_experiences()  # shape [mini_batch_size]
        (
            obs,
            actions,
            rewards,
            next_obs,
            done,
        ) = self.extract_transition_attributes_from_experiences(experiences)
        loss = self.compute_loss(obs, actions, rewards, next_obs, done)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_action_counts(Counter(actions.flatten().tolist()))

        if self.network_needs_updating():
            self.update_network()

    def update_action_counts(self, new_action_counts):

        for key, val in new_action_counts.items():
            self.action_counts[key] += val

    def compute_loss(self, obs, actions, rewards, next_obs, done):

        current_q_vals = self.calculate_current_q_values(obs, actions)
        target_q_vals = self.calculate_target_q_values(
            current_q_vals, rewards, next_obs, done
        )

        loss = torch.mean((target_q_vals - current_q_vals) ** 2)
        return loss

    def calculate_current_q_values(self, obs, actions):
        q_values = self.network(obs)
        actioned_q_values = self.calculate_actioned_q_values(q_values, actions)
        return actioned_q_values

    def calculate_target_q_values(self, current_q_vals, rewards, next_obs, done):

        next_q_vals = self.network(next_obs)
        target_q_vals_max = torch.max(next_q_vals, dim=-1).values

        # What should the Q values be updated to for the actions we took?
        target_q_vals = (
            current_q_vals * (1 - self.alpha)
            + self.alpha * rewards.flatten()
            + self.alpha * self.gamma * target_q_vals_max * (1 - done.int())
        )
        return target_q_vals

    @staticmethod
    def calculate_actioned_q_values(q_vals, actions):
        return q_vals[range(q_vals.shape[0]), actions.flatten()]

    @staticmethod
    def extract_transition_attributes_from_experiences(experiences):

        arr_to_tensor = lambda arr: torch.tensor(list(arr))  # noqa: E731

        extract_att = lambda att: arr_to_tensor(  # noqa: E731
            experiences[:, const.ATTRIBUTE_TO_INDEX[att]]
        )

        obs = extract_att("obs")
        actions = extract_att("actions")
        rewards = extract_att("rewards")
        next_obs = extract_att("next_obs")
        done = extract_att("done")

        return obs, actions, rewards, next_obs, done
