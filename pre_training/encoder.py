"""haakoas, matsjno"""

from torch import nn
import torchvision


class Encoder(nn.Module):
    """
    Class contaning the Simple Siamese network.
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.resnet = torchvision.models.resnet18()

    def forward(self, x):
        x = x[:, None, :, :]  # Add empty dimension to act as number of channels
        x = self.network(x)  # Expand number of channels to 3
        x = self.resnet(x)
        return x
