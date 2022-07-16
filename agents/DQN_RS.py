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
