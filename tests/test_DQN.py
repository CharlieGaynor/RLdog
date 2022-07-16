from get_agent import get_agent #  type: ignore

agent = get_agent()

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
