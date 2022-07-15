from agents.DQN import DQN
import gym
from networks.base_network import StandardNN

agent = DQN()

env = gym.make('CartPole-v1')

try:
    n_obs = env.reset().size
except AttributeError:
    pass

n_actions = env.action_space.n  # Not sure how to programatically get this - some manual config will do

network1 = StandardNN(n_obs, n_actions)
# agent = eg_model(network1, env.action_space.n)
buffer_size = 128
agent = DQN(network1, n_actions, env, buffer_size)

def test_transition_sampling():
    """
    Checks FI-FO works & buffer size is met 
    """
    for i in range(buffer_size):
        agent.transitions.appendleft(i)
    assert agent.transitions.pop() == 0
    agent.transitions.append(0)
    assert agent.transitions[-1] == 0

    agent.transitions.append(0)
    assert len(agent.transitions) == buffer_size