import random
class easy_env():

    def __init__(self):
        self.current_number = 0
        self.steps_taken = 0
        self.n_actions = 5
        self.n_obs = 1

    def reset(self):
        num = random.randint(1, 5)
        self.current_number  = num
        self.steps_taken = 0
        return num

    def step(self, num):
        self.steps_taken +=1
        done = 1 if self.steps_taken == 4 else 0
        reward = 1 if num == self.current_number else 0
        obs = random.randint(1, 5) if not done else -1

        return obs, reward, done , None
        