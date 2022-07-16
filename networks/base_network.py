import torch.nn as nn


class baseNN(nn.Module):
    """
    boring template
    """

    def __init__(self) -> None:
        super(baseNN, self).__init__()

    def forward(self, state):
        raise ValueError("Must implement forward method")
