from agents.DQN_based.DQN import DQN
import random
import numpy as np


class DQN_RS(DQN):
    """
    DQN but with random sampling instead. Use the same buffer,
    but sample randomly & don't delete experiences after you've sampled
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

        return np.array(random.sample(self.transitions, k=self.mini_batch_size), dtype=object)
