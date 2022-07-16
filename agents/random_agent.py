#  type: ignore
import random
from base_agent import agent_template


class random_agent(agent_template):
    """Random agent, who randomly picks one of the discrete options available"""

    def __init__(self, n_actions):
        """
        Args:
            n_obs (int): Dimensions of the state space (int for this project)
            n_actions (int): Number of possible actions
            lr (float, optional): Learning rate for the network. Defaults to 5e-4.
        """
        super().__init__()
        self.n_actions = n_actions

    def get_action(self, *kwargs, **args):
        """
        Return random action :D
        """
        return random.randint(0, self.n_actions - 1)

    def update_network(self, *kwargs, **args):
        pass
