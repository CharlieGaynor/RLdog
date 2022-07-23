from agents.DQN_based.DQN_TN import DQN_TN
import torch


class DDQN(DQN_TN):
    """A double DQN agent

    Now we use the policy network to find which action corresponds
    to the largest Q value for the 'next state'

    Then use the target network to get the Q value instead."""

    def __init__(self, config):
        DQN_TN.__init__(self, config)

    def calculate_target_q_values(
        self,
        current_q_vals: torch.Tensor,
        rewards: torch.FloatTensor,
        next_obs: torch.Tensor,
        done: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """
        Use the policy network to find which action corresponds
        to the largest Q value for the 'next state'
        """
        max_qvalue_actions = self.policy_network(next_obs).detach().argmax(1)
        target_q_vals_max = self.calculate_actioned_q_values(self.target_network(next_obs), max_qvalue_actions)

        # What the Q values should be updated to for the actions we took
        target_q_vals = (
            current_q_vals * (1 - self.alpha)
            + self.alpha * rewards.flatten()
            + self.alpha * self.gamma * target_q_vals_max * (1 - done.int())
        )
        return target_q_vals
