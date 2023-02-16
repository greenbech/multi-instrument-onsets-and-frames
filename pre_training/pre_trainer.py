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
        test_data = np.ones((5, 229, 229))
        test_data = torch.tensor(test_data).float()
        x = self.encoder(test_data)
        print(x)

    def train(self, epochs=10):
        test_data = torch.tensor(np.random.rand(10, 229, 229)).float()
        training_loader = DataLoader(test_data, batch_size=5, shuffle=True)
        optimizer = torch.optim.SGD(
            self.encoder.parameters(),
            lr=0.05,
            weight_decay=0.0001,
            momentum=0.9,
        )
        for i in range(epochs):
            running_loss = 0
            last_loss = 0
            for j, data in enumerate(training_loader):
                optimizer.zero_grad()
                # TODO: Apply random augmentations
                x1, x2 = data / 0.9, data * 0.9

                z1, z2 = self.encoder(x1), self.encoder(x2)
                print(z1)
                p1, p2 = self.predictor(z1), self.predictor(z2)

                loss = -(PreTrainer.cos_sim(p1, z2).mean() + PreTrainer.cos_sim(p2, z1).mean()) / 2
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if j % 2 == 1:
                    last_loss = running_loss / 1000
                    print(f"Batch {j+1} Loss: {last_loss}")
                    running_loss = 0

    @staticmethod
    def cos_sim(p, z):
        return torch.nn.CosineSimilarity(dim=1)(p, z)


def main():
    """
    Main function for running this python script.
    """
    pre_trainer = PreTrainer()
    pre_trainer.train()
    # pre_trainer.test()


if __name__ == "__main__":
    main()
