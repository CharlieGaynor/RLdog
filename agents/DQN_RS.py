from agents.DQN import DQN
import random
from typing import List, Any
import torch
from torch import nn
import numpy as np
from collections import deque
import gym
import constants as const
from collections import Counter
from networks import base_network


class DQN_RS(DQN):
    """
    DQN but with random sampling instead
    """

    def __init__(
        self,
        network: base_network.baseNN,
        n_actions: int,
        n_obs: int,
        env: gym.Env,
        max_games: int = 10000,
        games_to_decay_epsilon_for: int = 10000,
        lr: float = 1e-1,
        alpha: float = 0.01,
        gamma: float = 0.99,
        min_epsilon: float = 0.2,
        mini_batch_size=1,
        buffer_size=128,
        state_type: str = "DISCRETE",
    ):
        DQN.__init__(
            self,
            network,
            n_actions,
            n_obs,
            env,
            max_games,
            games_to_decay_epsilon_for,
            lr,
            alpha,
            gamma,
            min_epsilon,
            mini_batch_size,
            buffer_size,
            state_type,
        )

    def network_needs_updating(self) -> bool:
        """For DQN with random sampling, network needs updating every mini_batch steps"""
        return self.steps_without_update >= self.mini_batch_size

    def sample_experiences(self) -> np.ndarray:
        """
        Returns list of experiences with dimensions [mini_batch_size]
        """

        return np.array(
            random.sample(self.transitions, k=self.mini_batch_size), dtype=object
        )
