import gym
import torch
import numpy as np
from base_agent import agent_template
import sys


class executions:
    def __init__(self) -> None:
        pass

    @staticmethod
    def flatten_obs(*kwargs, **args):
        pass

    def execute_env(self, *kwargs, **args):
        pass


class basic_execution(executions):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def flatten_obs(observation, n_obs: int) -> torch.torch.FloatTensor:

        if isinstance(observation, int):
            indexes = [observation]
        else:
            indexes = [observation[0], observation[1] + 32, int(observation[2]) + 32 + 11]

        return torch.tensor([[1 if i in indexes else 0 for i in range(n_obs)]], dtype=torch.float32)

    def execute_env(
        self, agent: agent_template, env: gym.Env, n_epochs: int = 1000, train: bool = True, verbose: bool = True
    ) -> list:
        """
        Trains the {agent} for {epochs} many epochs

        args:
            agent (agent object): the agent object to train
            env (gym.make object): enviroment to use
            epochs (int): the number of epochs to train for
        returns:
            Not sure yet
        """
        rewards = []
        n_obs = 32 + 11 + 2  # env.observation_space.n
        actions = []

        for i in range(n_epochs):

            if (i % max((n_epochs // 5), 1) == 0) & (verbose):
                print("epoch:", i, end="")
                if hasattr(agent, "epsilon"):
                    print("Epsilon:", agent.epsilon, end="")
                print("\n")

            done = False
            observation_not_flat, info = env.reset(return_info=True)
            observation = self.flatten_obs(observation_not_flat, n_obs)

            # If espilon greedy then update it :D
            try:
                agent.update_epsilon(train)
            except AttributeError:
                pass
            while not done:
                action = agent.get_action(observation, train=train)
                next_observation_not_flat, reward, done, info = env.step(action)
                next_observation = self.flatten_obs(next_observation_not_flat, n_obs)

                # if (done) & (reward == 0):
                #     reward = -1
                # elif (done != 1) & (reward == 0):
                #     reward = -5e-4

                if train:
                    agent.update_network(
                        observation,
                        torch.tensor([action], dtype=torch.long),
                        torch.tensor([reward], dtype=torch.float32),
                        next_observation,
                        torch.tensor([done], dtype=torch.long),
                    )

                observation = next_observation
                rewards.append(reward)
                actions.append(action)
        return rewards, actions


class a2c_execution(executions):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def flatten_obs(*kwargs, **args):
        pass

    def execute_env(
        self, agent: agent_template, env: gym.Env, n_epochs: int = 1000, train: bool = True, verbose: bool = True
    ):

        num_inputs = 32 + 11 + 2
        num_outputs = 4

        actor_critic = agent()

        all_lengths = []
        average_lengths = []
        all_rewards = []
        entropy_term = 0

        for epoch in range(n_epochs):
            log_probs = []
            values = []
            rewards = []

            state = env.reset()
            while not done:
                value, policy_dist = actor_critic.forward(state)
                value = value.detach().numpy()[0, 0]
                dist = policy_dist.detach().numpy()

                action = np.random.choice(num_outputs, p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))
                new_state, reward, done, _ = env.step(action)

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                entropy_term += entropy
                state = new_state

                if done:
                    Qval, _ = actor_critic.forward(new_state)
                    Qval = Qval.detach().numpy()[0, 0]
                    all_rewards.append(np.sum(rewards))
                    all_lengths.append(steps)
                    average_lengths.append(np.mean(all_lengths[-10:]))
                    if epoch % 10 == 0:
                        sys.stdout.write(
                            "episode: {}, reward: {}, average length: {} \n".format(
                                epoch, np.sum(rewards), average_lengths[-1]
                            )
                        )

            # compute Q values
            Qvals = np.zeros_like(values)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + GAMMA * Qval
                Qvals[t] = Qval

            # update actor critic
            values = torch.FloatTensor(values)
            Qvals = torch.FloatTensor(Qvals)
            log_probs = torch.stack(log_probs)

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.001 * entropy_term


class policy_execution(executions):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def flatten_obs(observation, n_obs: int) -> torch.torch.FloatTensor:

        if isinstance(observation, int):
            indexes = [observation]
        else:
            indexes = [observation[0], observation[1] + 32, int(observation[2]) + 32 + 11]

        return torch.tensor([1 if i in indexes else 0 for i in range(n_obs)], dtype=torch.float32)

    def execute_env(
        self, agent: agent_template, env: gym.Env, n_epochs: int = 1000, train: bool = True, verbose: bool = True
    ) -> list:
        """
        Trains the {agent} for {epochs} many epochs

        args:
            agent (agent object): the agent object to train
            env (gym.make object): enviroment to use
            epochs (int): the number of epochs to train for
        returns:
            Not sure yet
        """
        rewards = []
        n_obs = 32 + 11 + 2  # env.observation_space.n
        actions = []

        for i in range(n_epochs):

            if (i % max((n_epochs // 5), 1) == 0) & (verbose):
                print("epoch:", i, end="")

            done = False
            transitions: list[tuple[torch.FloatTensor, int, float]] = []

            observation_not_flat, info = env.reset(return_info=True)
            observation = self.flatten_obs(observation_not_flat, n_obs)

            while not done:
                action = agent.get_action(observation, train=train)
                next_observation_not_flat, reward, done, info = env.step(action)
                next_observation = self.flatten_obs(next_observation_not_flat, n_obs)

                transitions.append((next_observation, action, reward))

                # if (done) & (reward == 0):
                #     reward = -1
                # elif (done != 1) & (reward == 0):
                #     reward = -5e-4
                if done:
                    if train:
                        agent.update_network(
                            torch.tensor(np.array([s.numpy() for (s, a, r) in transitions]), dtype=torch.float32),
                            torch.tensor([a for (s, a, r) in transitions], dtype=torch.float32),
                            torch.tensor([r for (s, a, r) in transitions], dtype=torch.float32),
                        )

                observation = next_observation
                rewards.append(reward)
                actions.append(action)
        return rewards, actions
