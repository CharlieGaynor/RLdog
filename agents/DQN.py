from typing import List, Any, Dict
import torch
from torch import nn
import numpy as np
from collections import deque
import constants as const
from collections import Counter
from networks.standard_network import standardNN
from networks.test_network import testNN
from tools.plotters import plot_results

# at the minute using epislon greedy - could generalise this out into a seperate class
# priority is having mini batches


class DQN(nn.Module):
    """Epsilon Greedy model

    Args:
        n_obs (int): Dimensions of the state space (int for this project)
        n_actions (int): Number of possible actions
        lr (float, optional): Learning rate for the network. Defaults to 5e-4.
        epsilon_decay (float, optional): Multiplication factor for epsilon
    """

    def __init__(self, config: Dict["str", Any]):

        super().__init__()

        self.metadata = config["metadata"]
        self.n_actions = self.metadata["n_actions"]
        self.n_obs = self.metadata["n_obs"]
        if self.metadata.get("test", False):
            self.network = testNN(self.n_obs, self.n_actions)
        else:
            self.network = standardNN(self.n_obs, self.n_actions)  # type: ignore

        self.env = self.metadata["env"]
        self.state_type = self.metadata["state_type"]

        self.hyperparameters = config["hyperparameters"]
        self.opt = torch.optim.Adam(
            self.network.parameters(), lr=self.hyperparameters["lr"]
        )
        self.epsilon: float = 1
        self.max_games = self.hyperparameters["max_games"]
        self.games_to_decay_epsilon_for = self.hyperparameters[
            "games_to_decay_epsilon_for"
        ]
        self.min_epsilon = self.hyperparameters["min_epsilon"]
        self.alpha = self.hyperparameters["alpha"]
        self.gamma = self.hyperparameters["gamma"]
        self.mini_batch_size = self.hyperparameters["mini_batch_size"]
        self.buffer_size = self.hyperparameters["buffer_size"]

        self.epsilon_decay = self.min_epsilon ** (1 / self.games_to_decay_epsilon_for)
        self.action_counts = {i: 0 for i in range(self.n_actions)}
        self.evaluation_action_counts = {i: 0 for i in range(self.n_actions)}
        self.state_is_discrete = self.state_type == "DISCRETE"

        self.transitions: deque[List[Any]] = deque([], maxlen=self.buffer_size)
        self.reward_averages: list[list[float]] = []
        self.evaluation_reward_averages: list[list[float]] = []
        self.evaluation_mode = False
        self.steps_without_update = 0
        self.games_played = 0

    def update_epsilon(self):
        if self.games_played < self.games_to_decay_epsilon_for:
            self.epsilon *= self.epsilon_decay

    def get_action(self, state: torch.Tensor):
        """Sample actions with epsilon-greedy policy"""

        q_values = self.network(state)

        if len(q_values) == 1:
            return q_values

        if self.evaluation_mode:
            return int(torch.argmax(q_values))

        ran_num = torch.rand(1)
        if ran_num < self.epsilon:
            return int(torch.randint(low=0, high=self.n_actions, size=(1,)))
        else:
            return int(torch.argmax(q_values))

    def _play_game(self) -> None:
        """Plays out one game"""
        next_obs_unformatted = np.array(self.env.reset())
        next_obs = self.format_obs(next_obs_unformatted)
        done = False
        rewards = []
        while not done:
            obs = next_obs
            action = self.get_action(obs)
            next_obs_unformatted, reward, done, termination, _ = self.env.step(action)
            next_obs = self.format_obs(np.array(next_obs_unformatted))
            rewards.append(reward)
            self.transitions.appendleft(
                [obs.tolist(), [action], [reward], next_obs.tolist(), [done]]
            )

            if termination:
                done = True
                reward *= 2

        self.update_epsilon()
        self.reward_averages.append([0.0, sum(rewards)])
        self.games_played += 1
        self.steps_without_update += len(rewards)

    def _evaluate_game(self):
        """
        Evaluates the models performance for one game
        """
        next_obs_unformatted = np.array(self.env.reset())
        next_obs = self.format_obs(next_obs_unformatted)
        done = False
        rewards = []
        actions = []
        while not done:
            obs = next_obs
            action = self.get_action(obs)
            next_obs_unformatted, reward, done, termination, _ = self.env.step(action)
            next_obs = self.format_obs(np.array(next_obs_unformatted))
            rewards.append(reward)
            actions.append(action)

            if termination:
                done = True
                reward *= 2

        self.evaluation_reward_averages.append([0.0, sum(rewards)])
        self.update_action_counts(Counter(actions))

    def format_obs(self, obs: np.ndarray) -> torch.Tensor:
        """format obs for optimal learning"""
        if self.state_is_discrete:
            encoded_state = np.zeros(self.n_obs, dtype=np.float32)
            encoded_state[obs.item()] = 1
            return torch.tensor(encoded_state, dtype=torch.float32)
        else:
            return torch.tensor(obs, dtype=torch.float32)

    def play_games(self, max_games: int = 0, verbose: bool = False) -> None:

        games_to_play = self.max_games if max_games == 0 else max_games

        if self.evaluation_mode:
            while games_to_play > 1:
                self._evaluate_game()
                games_to_play -= 1
            if verbose:
                total_rewards = [i[-1] for i in self.evaluation_reward_averages]
                plot_results(total_rewards)
                print("Action counts", self.evaluation_action_counts)
                print("Mean reward", sum(total_rewards) / len(total_rewards))

        else:
            while games_to_play > 1:
                self._play_game()
                if self.network_needs_updating():
                    self.update_network()
                games_to_play -= 1
            if verbose:
                total_rewards = [i[-1] for i in self.reward_averages]
                plot_results(total_rewards)

    def network_needs_updating(self) -> bool:
        """For standard DQN, network needs updated if self.transitions contains more than
        self.mini_batch_size items"""
        return len(self.transitions) >= self.mini_batch_size

    def sample_experiences(self) -> np.ndarray:
        """
        Returns list of experiences with dimensions [mini_batch_size]
        """
        experiences = []
        for _ in range(self.mini_batch_size):
            element = self.transitions.pop()
            experiences.append(element)

        return np.array(experiences, dtype=object)

    def update_network(self):

        experiences = self.sample_experiences()  # shape [mini_batch_size]
        (
            obs,
            actions,
            rewards,
            next_obs,
            done,
        ) = self.extract_transition_attributes_from_experiences(experiences)
        loss = self.compute_loss(obs, actions, rewards, next_obs, done)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_action_counts(Counter(actions.flatten().tolist()))

        if self.network_needs_updating():
            self.steps_without_update -= self.mini_batch_size
            self.update_network()

    def update_action_counts(self, new_action_counts):

        if self.evaluation_mode:
            for key, val in new_action_counts.items():
                self.evaluation_action_counts[key] += val
        else:
            for key, val in new_action_counts.items():
                self.action_counts[key] += val

    def compute_loss(self, obs, actions, rewards, next_obs, done):

        current_q_vals = self.calculate_current_q_values(obs, actions)
        target_q_vals = self.calculate_target_q_values(
            current_q_vals, rewards, next_obs, done
        )

        loss = torch.mean((target_q_vals - current_q_vals) ** 2)
        return loss

    def calculate_current_q_values(self, obs, actions):
        q_values = self.network(obs)
        actioned_q_values = self.calculate_actioned_q_values(q_values, actions)
        return actioned_q_values

    def calculate_target_q_values(self, current_q_vals, rewards, next_obs, done):

        next_q_vals = self.network(next_obs)
        target_q_vals_max = torch.max(next_q_vals, dim=-1).values

        # What should the Q values be updated to for the actions we took?
        target_q_vals = (
            current_q_vals * (1 - self.alpha)
            + self.alpha * rewards.flatten()
            + self.alpha * self.gamma * target_q_vals_max * (1 - done.int())
        )
        return target_q_vals

    @staticmethod
    def calculate_actioned_q_values(q_vals, actions):
        return q_vals[range(q_vals.shape[0]), actions.flatten()]

    @staticmethod
    def extract_transition_attributes_from_experiences(experiences):

        obs = experiences[:, const.ATTRIBUTE_TO_INDEX["obs"]].tolist()
        obs = torch.tensor(obs)

        actions = experiences[:, const.ATTRIBUTE_TO_INDEX["actions"]].tolist()
        actions = torch.tensor(actions)

        rewards = experiences[:, const.ATTRIBUTE_TO_INDEX["rewards"]].tolist()
        rewards = torch.tensor(rewards)

        next_obs = experiences[:, const.ATTRIBUTE_TO_INDEX["next_obs"]].tolist()
        next_obs = torch.tensor(next_obs)

        done = experiences[:, const.ATTRIBUTE_TO_INDEX["done"]].tolist()
        done = torch.tensor(done)

        return obs, actions, rewards, next_obs, done

    # @staticmethod
    # def numpy_array_to_torch_tensor
    #     arr_to_tensor = lambda arr: torch.tensor(list(arr))  # noqa: E731
