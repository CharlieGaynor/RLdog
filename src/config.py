import gym
from environments.easy import easy_env

config = {
    "BLACKJACK": {
        "DQN": {
            "hyperparameters": {
                "max_games": 10000,
                "min_epsilon": 0.1,
                "lr": 1e-2,
                "alpha": 0.9,
                "gamma": 0.99,
                "mini_batch_size": 12,
                "buffer_size": 120,
                "games_to_decay_epsilon_for": 10000 // 2,
                "tau": 0.01,
                "games_before_updating_target_network": 10,
                "hidden_size": 128,
            },
            "metadata": {
                "n_actions": 2,
                "n_obs": 3,
                "env": gym.make("Blackjack-v1", new_step_api=True),
                "state_type": "TUPLE",
            },
        },
        "POLICY": {
            "hyperparameters": {"max_games": 10000, "lr": 1e-2, "gamma": 0.95, "hidden_size": 128},
            "metadata": {
                "n_actions": 2,
                "n_obs": 32 + 11 + 2,
                "env": gym.make("Blackjack-v1", new_step_api=True),
                "state_type": "TUPLE",
                "one_hot_encoding_basepoints": [0, 32, 32 + 11],
            },
        },
        "AC": {
            "hyperparameters": {
                "max_games": 10000,
                "lr": 1e-2,
                "gamma": 0.95,
                "hidden_size": 128,
            },
            "metadata": {
                "n_actions": 2,
                "n_obs": 3,
                "env": gym.make("Blackjack-v1", new_step_api=True),
                "state_type": "TUPLE",
                "one_hot_encoding_basepoints": [],
            },
        },
    },
    "CARTPOLE": {
        "DQN": {
            "hyperparameters": {
                "max_games": 10000,
                "min_epsilon": 0.1,
                "lr": 1e-2,
                "alpha": 0.9,
                "gamma": 0.99,
                "mini_batch_size": 12,
                "buffer_size": 120,
                "games_to_decay_epsilon_for": 10000 // 2,
                "tau": 0.01,
                "games_before_updating_target_network": 10,
                "hidden_size": 128,
            },
            "metadata": {
                "n_actions": 2,
                "n_obs": 3,
                "env": gym.make("Blackjack-v1", new_step_api=True),
                "state_type": "TUPLE",
            },
        },
        "POLICY": {
            "hyperparameters": {"max_games": 10000, "lr": 1e-2, "gamma": 0.95, "hidden_size": 128},
            "metadata": {
                "n_actions": 2,
                "n_obs": 4,
                "env": gym.make("CartPole-v1", new_step_api=True),
                "state_type": "TUPLE",
                "one_hot_encoding_basepoints": [],
            },
        },
        "AC": {
            "hyperparameters": {"max_games": 10000, "lr": 1e-2, "gamma": 0.95, "hidden_size": 128},
            "metadata": {
                "n_actions": 2,
                "n_obs": 4,
                "env": gym.make("CartPole-v1", new_step_api=True),
                "state_type": "TUPLE",
                "one_hot_encoding_basepoints": [],
            },
        },
    },
    "TEST": {
        "DQN": {
            "hyperparameters": {
                "max_games": 1000,
                "min_epsilon": 0.1,
                "lr": 1e-3,
                "alpha": 0.1,
                "gamma": 0.99,
                "mini_batch_size": 10,
                "buffer_size": 20,
                "games_to_decay_epsilon_for": 750,
                "tau": 0.5,
                "games_before_updating_target_network": 10,
                "hidden_size": 128,
            },
            "metadata": {
                "n_actions": 5,
                "n_obs": 5,
                "env": easy_env(),  # type: ignore
                "state_type": "DISCRETE",
                "test": True,
            },
        },
        "REINFORCE": {
            "hyperparameters": {
                "max_games": 1000,
                "min_epsilon": 0.1,
                "lr": 1e-3,
                "alpha": 0.1,
                "gamma": 0.99,
                "mini_batch_size": 10,
                "buffer_size": 20,
                "games_to_decay_epsilon_for": 750,
                "tau": 0.5,
                "games_before_updating_target_network": 10,
                "hidden_size": 128,
            },
            "metadata": {
                "n_actions": 5,
                "n_obs": 5,
                "env": easy_env(),  # type: ignore
                "state_type": "DISCRETE",
                "test": True,
            },
        },
    },
}
