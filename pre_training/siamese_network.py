"""haakoas, matsjno"""

from torch import nn
import torchvision


class SiameseNetwork(nn.Module):
    """
    Class contaning the Simple Siamese network.
    """

    def __init__(self):
        super().__init__()
        self.network = torchvision.models.resnet18()

    def forward(self, x):
        x = self.network(x)
        return x
