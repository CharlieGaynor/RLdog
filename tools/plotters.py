from typing import Union
import numpy as np
from matplotlib import pyplot as plt  # type: ignore


def moving_average(x, window_length: Union[int, None] = None) -> np.ndarray:
    if window_length is None:
        window_length = max(len(x) // 50, 1)
    return np.convolve(x, np.ones(window_length), "valid") / window_length


def plot_results(test_rewards):
    ma_rewards = moving_average(test_rewards)
    plt.plot(ma_rewards, label="rewards")
    plt.show()
