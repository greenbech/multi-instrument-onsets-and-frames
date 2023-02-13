"""haakoas, matsjno"""

from siamese_network import SiameseNetwork
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np


class PreTrainer:
    """
    Pre-Training class for running a SimSiam pre-training procedure.
    """

    def __init__(self):
        self.output_size = 1000
        self.predictor_hidden_size = 2048
        self.encoder = SiameseNetwork()
        self.predictor = nn.Sequential(
            # Two fully connected layers creating a bottleneck
            nn.Linear(self.output_size, self.predictor_hidden_size),
            nn.ReLU(),
            nn.Linear(self.predictor_hidden_size, self.output_size),
            nn.Sigmoid(),
        )

    def test(self):
        test_data = np.random.rand(5, 3, 229, 229)
        test_data = torch.tensor(test_data).float()
        x = self.encoder(test_data)
        print(x)

    def train(self, epochs=5):
        test_data = torch.tensor(np.random.rand(10, 3, 229, 229)).float()
        training_loader = DataLoader(test_data, batch_size=5, shuffle=True)
        optimizer = torch.optim.SGD(
            self.encoder.parameters(),
            lr=0.05,
            weight_decay=0.0001,
            momentum=0.9,
        )
        for i in range(epochs):
            for i, data in enumerate(training_loader):
                optimizer.zero_grad()
                # TODO: Apply random augmentations
                x1, x2 = data, data * 0.9

                z1, z2 = self.encoder(x1), self.encoder(x2)
                p1, p2 = self.predictor(z1), self.predictor(z2)

                loss = PreTrainer.neg_cos_sim(p1, z2) / 2 + PreTrainer.neg_cos_sim(p2, z1) / 2
                loss.backward()
                optimizer.step()

    @staticmethod
    def neg_cos_sim(p, z):
        z = z.detach()
        p = torch.norm(p, dim=1)
        z = torch.norm(z, dim=1)
        print(f"p: {p.shape}")
        print(f"z: {z.shape}")
        print(f"pz: {(p*z).shape}")
        ehi = -(p * z).sum(dim=1).mean()
        print(ehi)
        return ehi


def main():
    """
    Main function for running this python script.
    """
    pre_trainer = PreTrainer()
    pre_trainer.train()
    # pre_trainer.test()


if __name__ == "__main__":
    main()
