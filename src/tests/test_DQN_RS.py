import random
from collections import Counter
import numpy as np
import config as config
from agents.DQN_based.DQN_RS import DQN_RS

test_config = config.config["TEST"]["DQN"]

agent = DQN_RS(test_config)
agent.epsilon = 0.5


def test_transition_sampling():
    """
    Checks FI-FO works & buffer size is met
    """
    buffer_size = agent.buffer_size
    mini_batch_size = agent.mini_batch_size
    for i in range(buffer_size):
        agent.transitions.appendleft(i)
    experience_sample: np.ndarray = agent.sample_experiences()

    assert len(experience_sample) == mini_batch_size  # Correct number of samples?
    assert len(agent.transitions) == buffer_size  # Did we remove any samples (shouldn't)


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
