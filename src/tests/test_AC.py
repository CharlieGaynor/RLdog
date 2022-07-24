from agents.Actor_critics.A2C import A2C
from config import config
import torch


def get_agent():
    test_config = config["CARTPOLE"]["AC"]
    return A2C(test_config)


def get_expected_discount(gamma):
    expected_discount_unscaled = [gamma - gamma**3, 1 - gamma**2, -gamma, -1]
    expected_discount = torch.tensor(expected_discount_unscaled)
    expected_discount = expected_discount / (expected_discount.std() + 1e-3)
    return expected_discount


def test_discounted_rewards():
    rewards = [0, 1, 0, -1]
    gamma = 1.0
    agent = get_agent()
    agent.gamma = gamma
    discounted_rewards = agent.compute_discounted_rewards(rewards)
    expected_discount = get_expected_discount(gamma)
    assert torch.equal(expected_discount, discounted_rewards)
