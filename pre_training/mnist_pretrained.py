# MNIST(root[, train, transform, ...])

import torchvision
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.dataloader import default_collate

from mnist_encoder import MnistEncoder


class Net(nn.Module):
    def __init__(self, device):
        super(Net, self).__init__()
        self.encoder = MnistEncoder().to(device)
        self.encoder.load_state_dict(torch.load("mnist_encoder.pt", map_location=device))
        self.encoder.eval()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.encoder(x, False)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.nll_loss(outputs, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(inputs),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    """
    Main function for running this python script.
    """
    # device = torch.device("cuda:0")
    device = torch.device("cpu")

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_train = torchvision.datasets.MNIST("pre_training/data/", train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.MNIST("pre_training/data/", train=False, download=True, transform=transform)

    indices = torch.arange(0, 10000)
    mnist_train = torch.utils.data.Subset(mnist_train, indices)
    mnist_test = torch.utils.data.Subset(mnist_test, indices)

    batch_size = 64
    training_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )
    test_loader = torch.utils.data.DataLoader(
        mnist_test,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)),
    )

    print(f"Cuda: {torch.cuda.is_available()}")
    # print(f"Device: {torch.cuda.get_device_name(0)}")
    model = Net(device).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    epochs = 20
    for epoch in range(1, epochs + 1):
        train(model, training_loader, optimizer, epoch)
        test(model, test_loader)
        scheduler.step()

    if True:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
