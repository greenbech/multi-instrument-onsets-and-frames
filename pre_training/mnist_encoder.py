"""haakoas, matsjno"""

from torch import nn
import torch
import numpy as np


class MnistEncoder(nn.Module):
    """
    Class contaning the Simple Siamese network.
    """

    def __init__(self):
        super().__init__()
        self.pre_trainer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.pre_trainer(x)
        x = torch.flatten(x, 1)
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)


def test():
    mnist = MnistEncoder()
    test_data = np.ones((5, 1, 229, 229))
    test_data = torch.tensor(test_data).float()
    x = mnist(test_data)
    print(x)
    print(x.shape)


if __name__ == "__main__":
    test()
