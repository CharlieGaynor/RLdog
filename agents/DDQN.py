from agents.DQN_TN import DQN_TN
import torch


class DDQN(DQN_TN):
    """A double DQN agent"""

    agent_name = "DDQN"

    def __init__(self, config):
        DQN_TN.__init__(self, config)

    def calculate_target_q_values(self, current_q_vals, rewards, next_obs, done):

        max_qvalue_actions = self.policy_network(next_obs).detach().argmax(1)
        target_q_vals_max = self.calculate_actioned_q_values(
            self.target_network(next_obs), max_qvalue_actions
        )

        # What should the Q values be updated to for the actions we took?
        target_q_vals = (
            current_q_vals * (1 - self.alpha)
            + self.alpha * rewards.flatten()
            + self.alpha * self.gamma * target_q_vals_max * (1 - done.int())
        )
        return target_q_vals
