from agents.DQN import DQN
import gym
from networks.base_network import StandardNN


def grab_agent() -> DQN:
    """
    Instantiates agent for testing
    """

    #  Use the new step API, which returns a truncated observation
    env = gym.make("CartPole-v1", new_step_api=True)

    try:
        n_obs = env.reset().size  # type: ignore
    except AttributeError:
        pass

    n_actions = (
        env.action_space.n  # type: ignore
    )  # Not sure how to programatically get this - some manual config will do

    network1 = StandardNN(n_obs, n_actions)
    # agent = eg_model(network1, env.action_space.n)
    buffer_size = 128
    agent = DQN(network1, n_actions, env, buffer_size)

    return agent
