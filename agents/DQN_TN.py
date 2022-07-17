from agents.DQN import DQN
import random
import numpy as np
from networks.standard_network import standardNN
from networks.test_network import testNN
from collections import Counter
import torch


class DQN_TN(DQN):
    """
    DQN but with Target Network implemented instead
    """

    def __init__(self, config):
        DQN.__init__(self, config)
        self.policy_network = testNN(self.n_obs, self.n_actions)
        self.target_network = testNN(self.n_obs, self.n_actions)
        self.target_network.eval()

        self.opt = torch.optim.Adam(
            self.policy_network.parameters(), lr=self.hyperparameters["lr"]
        )
        self.copy_model_over(
            from_model=self.policy_network, to_model=self.target_network
        )
        self.consecutive_updates = 0

        hyperparameters = config["hyperparameters"]
        self.games_before_updating_target_network = hyperparameters[
            "games_before_updating_target_network"
        ]
        self.tau = 0.1  # hyperparameters['tau']

    def update_network(self):

        experiences = self.sample_experiences()  # shape [mini_batch_size]
        obs, actions, rewards, next_obs, done = self.attributes_from_experiences(
            experiences
        )
        loss = self.compute_loss(obs, actions, rewards, next_obs, done)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.consecutive_updates += 1

        if self.consecutive_updates >= self.games_before_updating_target_network:
            self.consecutive_updates = 0

        self.update_action_counts(Counter(actions.flatten().tolist()))

        self.copy_model_over(
            from_model=self.policy_network, to_model=self.target_network
        )
        # self.soft_update_of_target_network()

        if self.network_needs_updating():
            self.steps_without_update -= self.mini_batch_size
            self.update_network()

    def soft_update_of_target_network(self):
        """Updates the target network in the direction of the local network but by taking a small step size instead
        The target network's parameter values trail the local networks. This helps stabilise training apparently"""

        for to_network, from_network in zip(
            self.target_network.parameters(), self.policy_network.parameters()
        ):
            to_network.data.copy_(
                (1.0 - self.tau) * to_network.data.clone()
                + self.tau * from_network.data.clone()
            )

    def calculate_target_q_values(self, current_q_vals, rewards, next_obs, done):
        """Computes the target q values for the actions we took"""

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
    def copy_model_over(to_model, from_model):
        """Copies model parameters from from_model to to_model"""
        to_model.load_state_dict(from_model.state_dict())
