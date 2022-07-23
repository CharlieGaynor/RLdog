import random
import gym


class easy_env(gym.Env):
    """
    Simple environment of a basic game for testing. The game is:
    The environment gives a random number [1, 5], 4 times
    the agent simply has to give the same number back as each timestep
    Reward = 1 if they do, else 0
    """

    def __init__(self):
        self.current_number = 0
        self.steps_taken = 0
        self.n_actions = 5
        self.n_obs = 5

    def reset(self):
        num = random.randint(0, 4)
        self.current_number = num
        self.steps_taken = 0
        return num

    def step(self, num):
        self.steps_taken += 1
        done = 1 if self.steps_taken == 4 else 0
        reward = 1.0 if num == self.current_number else 0.0

        num = random.randint(0, 4)
        self.current_number = num

        return num, reward, done, False, ""
