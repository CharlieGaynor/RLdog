import config
from typing import Dict, Any
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
