import random
from collections import Counter
import numpy as np
import config as config
from agents.DQN_based.DQN import DQN

test_config = config.config["TEST"]["DQN"]

agent = DQN(test_config)
agent.epsilon = 0.5


def test_transition_sampling():
    """
    Checks FI-FO works & buffer size is met
    """
    buffer_size = agent.buffer_size
    for i in range(buffer_size):
        agent.transitions.appendleft(i)
    assert agent.transitions.pop() == 0
    agent.transitions.append(0)
    assert agent.transitions[-1] == 0

    agent.transitions.append(0)
    assert len(agent.transitions) == buffer_size


def test_get_action():
    """
    Tests we grab an action correctly, including using Epsilon greedy right
    """
    actions = []
    agent.epsilon = 0
    for _ in range(10):
        obs = 1
        obs_arr = np.array(obs)
        obs_f = agent.format_obs(obs_arr)
        actions.append(agent.get_action(obs_f))
    assert max(dict(Counter(actions)).values()) == 10

    actions = []
    agent.epsilon = 1
    for _ in range(100):
        num = random.randint(0, 4)
        actions.append(agent.get_action(agent.format_obs(np.array(num))))

    assert max(dict(Counter(actions)).values()) < 50
    agent.epsilon = 0.5
