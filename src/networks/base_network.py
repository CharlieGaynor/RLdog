import torch.nn as nn
from abc import ABC, abstractmethod


class baseNN(nn.Module, ABC):
    """
    boring template
    """

    def __init__(self) -> None:
        nn.Module.__init__(self)

    @abstractmethod
    def forward(self, state):
        pass
