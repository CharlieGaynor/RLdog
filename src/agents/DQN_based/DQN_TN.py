from typing import Any, Dict
from agents.DQN_based.DQN import DQN
from networks.base_network import baseNN
from networks.DQN_network import dqnNN
from collections import Counter
import torch

from networks.test_network import testNN


class DQN_TN(DQN):
    """
    DQN but with Target Network implemented instead

    This means we use an old target network to esimate the 'next' Q values
    This prevents too much guess work and stabilises training.

    Everytime we train, update the target network by moving a small step in the
    direction of the policy network
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        DQN.__init__(self, config)

        if self.metadata.get("test", False):
            self.target_network = testNN(self.n_obs, self.n_actions)
        else:
            self.target_network = dqnNN(self.n_obs, self.n_actions, self.hidden_size)  # type:ignore
        self.target_network.eval()  # We don't ever train this model, so just evaluate

        # Start off with the same networks
        self.copy_network_over(from_network=self.policy_network, to_network=self.target_network)
        self.tau: float = self.hyperparameters["tau"]

    def update_network(self) -> None:

        experiences = self.sample_experiences()  # shape [mini_batch_size]
        obs, actions, rewards, next_obs, done = self.attributes_from_experiences(experiences)
        loss = self.compute_loss(obs, actions, rewards, next_obs, done)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_action_counts(Counter(actions.flatten().tolist()))
        self.soft_update_of_target_network()

        if self.network_needs_updating():
            self.steps_without_update -= self.mini_batch_size
            self.update_network()

    def soft_update_of_target_network(self):
        """Updates the target network in the direction of the local network,
                but by taking a small step size instead, with tau << 1
        .
                This helps stabilise training apparently"""

        for to_network, from_network in zip(self.target_network.parameters(), self.policy_network.parameters()):
            to_network.data.copy_((1.0 - self.tau) * to_network.data.clone() + self.tau * from_network.data.clone())

    def calculate_target_q_values(
        self,
        current_q_vals: torch.Tensor,
        rewards: torch.FloatTensor,
        next_obs: torch.Tensor,
        done: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Uses the target network to calculate the next_q_values instead"""

        next_q_vals = self.target_network(next_obs).detach()
        target_q_vals_max = torch.max(next_q_vals, dim=-1).values

        # What should the Q values be updated to for the actions we took?
        target_q_vals = (
            current_q_vals * (1 - self.alpha)
            + self.alpha * rewards.flatten()
            + self.alpha * self.gamma * target_q_vals_max * (1 - done.int())
        )
        return target_q_vals

    @staticmethod
    def copy_network_over(to_network: baseNN, from_network: baseNN) -> None:
        """Copies model parameters from from_model to to_model"""
        to_network.load_state_dict(from_network.state_dict())
