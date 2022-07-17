from agents.DQN import DQN

import random
import numpy as np


class DQN_PR(DQN):
    """
    DQN but with Prioritised replay instead
    """

    def __init__(self, config):
        DQN.__init__(self, config)
        

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
