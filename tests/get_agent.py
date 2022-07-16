from agents.DQN import DQN
from networks.test_network import testNN
from environments.easy import easy_env


def grab_agent() -> DQN:
    """
    Instantiates agent for testing
    """

    #  Use the new step API, which returns a truncated observation
    env = easy_env()

    n_obs = env.n_obs

    n_actions = env.n_actions

    network1 = testNN(n_obs, n_actions)
    # agent = eg_model(network1, env.action_space.n)
    buffer_size = 128
    agent = DQN(network1, n_actions, n_obs, env, buffer_size)
    agent.epsilon = 0.5

    return agent
