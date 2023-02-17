import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np


def main():
    """
    Main function for running this python script.
    """
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))]
    )

    mnist_train = torchvision.datasets.MNIST("pre_training/data/", train=True, download=True, transform=transform)

    indices = torch.arange(10, 20)
    mnist_train = torch.utils.data.Subset(mnist_train, indices)

    batch_size = 5
    training_loader = torch.utils.data.DataLoader(
        mnist_train,
        batch_size=batch_size,
        shuffle=False,
    )

    fill = -0.42421296
    augmenter = torchvision.transforms.GaussianBlur(5, (1.0, 2.0))
    augmenter = torchvision.transforms.CenterCrop()
    resizer = torchvision.transforms.Resize(28)
    augmenter = torchvision.transforms.RandomRotation((-90, 90), fill=fill)
    augmenter = torchvision.transforms.RandomPerspective(distortion_scale=0.5, fill=fill)

    for j, (data, _) in enumerate(training_loader):
        augmented_data = augmenter(data)
        augmented_data = resizer(augmented_data)
        np_original_data = np.array(data)
        np_augmented_data = np.array(augmented_data)
        fig, (axs1, axs2) = plt.subplots(2, 5)
        for i, ax in enumerate(axs1):
            ax.imshow(np_original_data[i].squeeze(), cmap="gray")
        for i, ax in enumerate(axs2):
            ax.imshow(np_augmented_data[i].squeeze(), cmap="gray")
        plt.show()


if __name__ == "__main__":
    main()
