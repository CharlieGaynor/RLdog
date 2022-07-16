import random
from tests.get_agent import grab_agent
import torch
from collections import Counter

agent = grab_agent()


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
        actions.append(agent.get_action(torch.tensor([1], dtype=torch.float32)))
    assert Counter(actions)[1] == 10

    actions = []
    agent.epsilon = 1
    for _ in range(100):
        actions.append(
            agent.get_action(torch.tensor([random.randint(1, 5)], dtype=torch.float32))
        )

    assert Counter(actions)[1] < 50
    agent.epsilon = 0.5
