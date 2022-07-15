import gym 
import constants as const
from networks.base_network import StandardNN
from agents.DQN import DQN
if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    try:
        n_obs = env.reset().size
    except AttributeError:
        pass

    n_actions = env.action_space.n  # Not sure how to programatically get this - some manual config will do

    network1 = StandardNN(n_obs, n_actions)
    # agent = eg_model(network1, env.action_space.n)
    agent = DQN(network1, n_actions, env)
    agent.play_games(100)