import config
import gym
from agents.DQN import DQN
from networks.standard_network import standardNN
from networks.basic_network import basicNN
import gym
from typing import Union, Dict, Any
import numpy as np
from environments.easy import easy_env
from agents.DQN_RS import DQN_RS

bj_config: Dict[Any, Any] = config.config["BLACKJACK"]
bj_config["hyperparameters"]["max_games"] = 2000
bj_config["hyperparameters"]["lr"] = 1e-3
bj_config["hyperparameters"]["alpha"] = 0.05

# agent = DQN(bj_config)
# agent.play_games(verbose=True)
# agent.evaluation_mode = True
# agent.play_games(1000, verbose=True)
# agent.evaluation_mode = False

agent = DQN_RS(bj_config)
agent.play_games(verbose=True)
# agent.evaluation_mode = True
# agent.play_games(5000, verbose=True)
# agent.evaluation_mode = False
