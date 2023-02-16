"""haakoas, matsjno"""

from torch import nn
import torchvision


class MnistEncoder(nn.Module):
    """
    Class contaning the Simple Siamese network.
    """

    def __init__(self, layer_size):
        super().__init__()
        self.pre_trainer = nn.Sequential(
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 10),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x[:, None, :, :]  # Add empty dimension to act as number of channels
        x = self.network(x)  # Expand number of channels to 3
        x = self.resnet(x)
        return x


def main():
    mnist = MnistEncoder(1000)
