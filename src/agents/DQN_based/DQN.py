from typing import List, Any, Dict, Tuple
import torch
from torch import nn
import numpy as np
from collections import deque
import constants as const
from collections import Counter
from networks.DQN_network import dqnNN
from networks.test_network import testNN
from tools.plotters import plot_results

# at the minute using epislon greedy - could generalise this out into a seperate class
# priority is having mini batches


class DQN(nn.Module):
    """
    Deep Q-learning agent, capable of learning in a variety of envrionments

    args:
        config (Dict[str, Any]): Bunch of hyperparameters and necessary metadata for each
        environment.

    """

    def __init__(self, config: Dict[str, Any]):

        super().__init__()

        self.metadata = config["metadata"]
        self.n_actions = self.metadata["n_actions"]
        self.n_obs = self.metadata["n_obs"]
        if self.metadata.get("test", False):
            self.policy_network = testNN(self.n_obs, self.n_actions)
        else:
            self.policy_network = dqnNN(self.n_obs, self.n_actions)  # type: ignore
        self.env = self.metadata["env"]
        self.state_type = self.metadata["state_type"]

        self.hyperparameters = config["hyperparameters"]
        self.opt = torch.optim.Adam(self.policy_network.parameters(), lr=self.hyperparameters["lr"])
        self.epsilon = 1.0
        self.max_games: int = self.hyperparameters["max_games"]
        self.games_to_decay_epsilon_for: int = self.hyperparameters["games_to_decay_epsilon_for"]
        self.min_epsilon: float = self.hyperparameters["min_epsilon"]
        self.alpha: float = self.hyperparameters["alpha"]
        self.gamma: float = self.hyperparameters["gamma"]
        self.mini_batch_size: int = self.hyperparameters["mini_batch_size"]
        self.buffer_size: int = self.hyperparameters["buffer_size"]

        self.epsilon_decay: float = self.min_epsilon ** (1 / self.games_to_decay_epsilon_for)
        self.action_counts = {i: 0 for i in range(self.n_actions)}
        self.evaluation_action_counts = {i: 0 for i in range(self.n_actions)}
        self.state_is_discrete: bool = self.state_type == "DISCRETE"

        self.transitions: deque[List[Any]] = deque([], maxlen=self.buffer_size)
        self.reward_averages: list[list[float]] = []
        self.evaluation_reward_averages: list[list[float]] = []
        self.evaluation_mode = False
        self.steps_without_update = 0
        self.games_played = 0

    def _play_game(self) -> None:
        """
        Interact with the environment until 'done'
        store transitions in self.transitions & updates
        epsilon after each game has finished
        """
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
            self.transitions.appendleft([obs.tolist(), [action], [reward], next_obs.tolist(), [done]])

            if termination:
                done = True

        self.update_epsilon()
        self.reward_averages.append([0.0, sum(rewards)])
        self.steps_without_update += len(rewards)

    def get_action(self, state: torch.Tensor) -> int:
        """Sample actions with epsilon-greedy policy
        use epsilon = 0 for evaluation mode"""

        with torch.no_grad():
            q_values = self.policy_network(state)

        # If we have regression-like problem
        if len(q_values) == 1:
            return q_values

        if not self.evaluation_mode:
            ran_num = torch.rand(1)
            if ran_num < self.epsilon:
                return int(torch.randint(low=0, high=self.n_actions, size=(1,)))

        return int(torch.argmax(q_values))

    def update_network(self):
        """
        Sample experiences, compute & back propagate loss,
        and call itself recursively if there are still samples to train on
        """

        experiences = self.sample_experiences()  # shape [mini_batch_size]
        (
            obs,
            actions,
            rewards,
            next_obs,
            done,
        ) = self.attributes_from_experiences(experiences)
        loss = self.compute_loss(obs, actions, rewards, next_obs, done)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.update_action_counts(Counter(actions.flatten().tolist()))

        if self.network_needs_updating():
            self.steps_without_update -= self.mini_batch_size
            self.update_network()

    def format_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Allow obs to be passed into pytorch model"""
        if self.state_is_discrete:
            encoded_state = np.zeros(self.n_obs, dtype=np.float32)
            encoded_state[obs.item()] = 1
            return torch.tensor(encoded_state, dtype=torch.float32)
        else:
            return torch.tensor(obs, dtype=torch.float32)

    def network_needs_updating(self) -> bool:
        """
        For standard DQN, network needs updated if self.transitions contains more than
        self.mini_batch_size items
        """
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

    def update_action_counts(self, new_action_counts: Dict[int, int]) -> None:

        if self.evaluation_mode:
            for key, val in new_action_counts.items():
                self.evaluation_action_counts[key] += val
        else:
            for key, val in new_action_counts.items():
                self.action_counts[key] += val

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.LongTensor,
        rewards: torch.FloatTensor,
        next_obs: torch.Tensor,
        done: torch.BoolTensor,
    ) -> torch.Tensor:
        """Compute loss according to Q learning equation, and return MSE"""

        current_q_vals = self.calculate_current_q_values(obs, actions)
        with torch.no_grad():
            target_q_vals = self.calculate_target_q_values(current_q_vals, rewards, next_obs, done)

        loss = torch.mean((target_q_vals - current_q_vals) ** 2)
        return loss

    def calculate_current_q_values(self, obs: torch.Tensor, actions: torch.LongTensor) -> torch.Tensor:
        """Computes the Q values for the actions we took"""
        q_values = self.policy_network(obs)
        actioned_q_values = self.calculate_actioned_q_values(q_values, actions)
        return actioned_q_values

    def calculate_target_q_values(
        self,
        current_q_vals: torch.Tensor,
        rewards: torch.FloatTensor,
        next_obs: torch.Tensor,
        done: torch.BoolTensor,
    ) -> torch.FloatTensor:
        """Computes the 'Target' Q values for the actions we took
        Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max_a'{Q(s',a')} )"""

        next_q_vals = self.policy_network(next_obs).detach()
        target_q_vals_max = torch.max(next_q_vals, dim=-1).values

        # What should the Q values be updated to for the actions we took?
        target_q_vals = (
            current_q_vals * (1 - self.alpha)
            + self.alpha * rewards.flatten()
            + self.alpha * self.gamma * target_q_vals_max * (1 - done.int())
        )
        return target_q_vals

    @staticmethod
    def calculate_actioned_q_values(q_vals: torch.FloatTensor, actions: torch.LongTensor) -> torch.Tensor:
        """Give me Q values for all actions, and the actions you took.
        I will return you only the Q values for the actions you took
        """
        return q_vals[range(q_vals.shape[0]), actions.flatten()]

    def play_games(self, games_to_play: int = 0, verbose: bool = False) -> None:
        """
        Play the games, updating at each step the network if not self.evaluation_mode

        Verbose mode shows some stats at the end of the training, and a graph.

        You're welcome.
        """

        games_to_play = self.max_games if games_to_play == 0 else games_to_play

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
                self.games_played += 1
                if self.network_needs_updating():
                    self.update_network()
                games_to_play -= 1
            if verbose:
                total_rewards = [i[-1] for i in self.reward_averages]
                plot_results(total_rewards)

    def _evaluate_game(self) -> None:
        """
        Evaluates the models performance for one game. Seperate function as this
        runs quicker, at the price of not storing transitions.

        Runs when self.evaluation_mode = True
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

        self.evaluation_reward_averages.append([0.0, sum(rewards)])
        self.update_action_counts(Counter(actions))

    def update_epsilon(self) -> None:
        """Slowly decrease epsilon to reduce exploration"""
        if self.games_played < self.games_to_decay_epsilon_for:
            self.epsilon *= self.epsilon_decay

    @staticmethod
    def attributes_from_experiences(
        experiences: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.LongTensor, torch.FloatTensor, torch.Tensor, torch.BoolTensor,]:
        """Extracts, transforms (and loads, hehe)
        the attributes hidden in within experiences"""

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
