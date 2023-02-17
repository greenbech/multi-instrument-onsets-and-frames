"""haakoas, matsjno"""

import torch
import torchvision
from torch import nn
import numpy as np
from torch.utils.data.dataloader import default_collate

# from siamese_network import SiameseNetwork
from mnist_encoder import MnistEncoder


class PreTrainer:
    """
    Pre-Training class for running a SimSiam pre-training procedure.
    """

    def __init__(self, device):
        self.output_size = 28 * 28
        self.predictor_hidden_size = 2048
        self.encoder = MnistEncoder().to(device)
        self.predictor = nn.Sequential(
            # Two fully connected layers creating a bottleneck
            nn.Linear(self.output_size, self.predictor_hidden_size),
            nn.ReLU(),
            nn.Linear(self.predictor_hidden_size, self.output_size),
            nn.Sigmoid(),
        ).to(device)

    def test(self):
        test_data = np.ones((5, 229, 229))
        test_data = torch.tensor(test_data).float()
        x = self.encoder(test_data)
        print(x)

    def train(self, training_loader, epochs=10):
        # test_data = torch.tensor(np.random.rand(10, 229, 229)).float()
        # training_loader = DataLoader(test_data, batch_size=5, shuffle=True)
        optimizer = torch.optim.SGD(
            self.encoder.parameters(),
            lr=0.025,
            weight_decay=0.0001,
            momentum=0.9,
        )
        for i in range(epochs):
            running_loss = 0
            last_loss = 0
            for j, (data, _) in enumerate(training_loader):
                optimizer.zero_grad()
                x1, aug_idx = PreTrainer.rand_augment(data)
                x2, _ = PreTrainer.rand_augment(data, avoid_idx=aug_idx)

                z1, z2 = self.encoder(x1), self.encoder(x2)
                p1, p2 = self.predictor(z1), self.predictor(z2)

                loss = -(PreTrainer.cos_sim(p1, z2).mean() + PreTrainer.cos_sim(p2, z1).mean()) / 2
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if j % 2 == 1:
                    last_loss = running_loss / 1000
                    print(f"Batch {j+1} Loss: {last_loss}")
                    running_loss = 0

    def save(self, path: str):
        self.encoder.save(path)

    @staticmethod
    def cos_sim(p, z):
        return torch.nn.CosineSimilarity(dim=1)(p, z)

    @staticmethod
    def rand_augment(data, avoid_idx=-1):
        fill = -0.42421296
        aug_idxs = [0, 1, 2, 3]
        if avoid_idx in range(0, len(aug_idxs)):
            aug_idxs.pop(avoid_idx)
        rand_num = np.random.randint(0, len(aug_idxs))
        aug_idx = aug_idxs[rand_num]
        if aug_idx == 0:
            # Blur image
            augmenter = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=(1.0, 2.0))
            return augmenter(data), aug_idx
        elif aug_idx == 1:
            # Zoom in on image, keep dimensions
            rand_crop = np.random.randint(20, 24)
            augmenter = torchvision.transforms.CenterCrop(size=rand_crop)
            resizer = torchvision.transforms.Resize(size=28)
            return resizer(augmenter(data)), aug_idx
        elif aug_idx == 2:
            # Rotate image random angle
            augmenter = torchvision.transforms.RandomRotation(degrees=(-90, 90), fill=fill)
            return augmenter(data), aug_idx
        elif aug_idx == 3:
            # Apply random perspective to image
            augmenter = torchvision.transforms.RandomPerspective(distortion_scale=0.5, fill=fill)
            return augmenter(data), aug_idx


def main():
    """
    Main function for running this python script.
    """
    device = torch.device("cuda:0")

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_train = torchvision.datasets.MNIST("pre_training/data/", train=True, download=True, transform=transform)

    indices = torch.arange(10000, 60000)
    mnist_train = torch.utils.data.Subset(mnist_train, indices)

    batch_size = 512
    training_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )

    print(f"Cuda: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    pre_trainer = PreTrainer(device)
    pre_trainer.train(training_loader, epochs=3)
    pre_trainer.save("mnist_encoder.pt")


main()
