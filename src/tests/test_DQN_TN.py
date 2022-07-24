import config as config
from agents.DQN_based.DQN_TN import DQN_TN


test_config = config.config["TEST"]["DQN"]

agent = DQN_TN(test_config)
agent.epsilon = 0.5


def test_copy_model():

    agent.target_network.l1.weight.data.fill_(2)

    state_a = agent.policy_network.state_dict().__str__()
    state_b = agent.target_network.state_dict().__str__()
    assert state_a != state_b
    agent.copy_network_over(from_network=agent.policy_network, to_network=agent.target_network)
    state_b = agent.target_network.state_dict().__str__()
    assert state_a == state_b


def test_soft_update():
    agent.policy_network.l1.weight.data.fill_(5)
    state_b = agent.target_network.state_dict().__str__()
    agent.soft_update_of_target_network()
    state_c = agent.target_network.state_dict().__str__()
    assert state_c != state_b
