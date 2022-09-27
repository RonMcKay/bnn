import bnn
from bnn.utils import BayesNetWrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as trans

EPOCHS = 200
BATCH_SIZE = 256
NUM_WORKERS = 0
GPU = 0
device = None  # will be set in main


class FullyBayesianNeuralNetwork(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()

        self.model = bnn.Sequential(
            bnn.BConv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.Softplus(),
            bnn.BConv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.Softplus(),
            bnn.BConv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.Softplus(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            bnn.BLinear(in_features=4 * 4 * 128, out_features=64),
            nn.Softplus(),
            bnn.BLinear(in_features=64, out_features=32),
            nn.Softplus(),
            bnn.BLinear(in_features=32, out_features=n_classes),
        )

        # self.model = bnn.Sequential(
        #     nn.Flatten(),
        #     bnn.BLinear(in_features=28 * 28, out_features=512),
        #     nn.Softplus(),
        #     bnn.BLinear(in_features=512, out_features=256),
        #     nn.Softplus(),
        #     bnn.BLinear(in_features=256, out_features=128),
        #     nn.Softplus(),
        #     bnn.BLinear(in_features=128, out_features=64),
        #     nn.Softplus(),
        #     bnn.BLinear(in_features=64, out_features=32),
        #     nn.Softplus(),
        #     bnn.BLinear(in_features=32, out_features=n_classes)
        # )

    def forward(self, x):
        out, kl = self.model(x)
        return out, kl


def main():
    global device
    if torch.cuda.is_available() and GPU is not None:
        device = torch.device("cuda:{}".format(GPU))
    else:
        print("Cuda is not available. Training will be done on CPU.")
        device = torch.device("cpu")

    train_data = torchvision.datasets.EMNIST(
        root="/data/datasets/cl",
        split="digits",
        train=True,
        download=True,
        transform=trans.ToTensor(),
    )
    # take only a subset of 10000 samples for a faster example
    train_data = torch.utils.data.Subset(
        train_data, list(torch.randperm(len(train_data)).numpy()[:10000])
    )
    val_data = torchvision.datasets.EMNIST(
        root="/data/datasets/cl",
        split="digits",
        train=False,
        download=True,
        transform=trans.ToTensor(),
    )
    val_data = torch.utils.data.Subset(
        val_data, list(torch.randperm(len(val_data)).numpy()[:10000])
    )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    net = FullyBayesianNeuralNetwork()
    net = net.to(device)

    net_wrapper = BayesNetWrapper(
        net=net,
        learning_rate=1e-3,
        cuda=True if GPU is not None else False,
        scheduling=True,
        device_ids=[GPU] if GPU is not None else GPU,
    )

    for epoch in range(EPOCHS):
        avg_train_loss = 0
        avg_train_acc = 0
        avg_val_acc = 0

        # Training Phase
        for i, (x, y) in enumerate(train_loader):
            # for other ways of setting the batch_weight see e.g. 'get_beta' in
            # https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/metrics.py
            # or the corresponding publication.
            batch_weight = 2 ** (len(train_loader) - (i + 1)) / (
                2 ** len(train_loader) - 1
            )  # Blundell KL weights
            loss, acc = net_wrapper.fit(x, y, batch_weight=batch_weight, samples=1)
            avg_train_loss += loss
            avg_train_acc += acc

        net_wrapper.scheduler.step()

        avg_train_loss /= len(train_loader)
        avg_train_acc /= len(train_loader)

        # Validation Phase
        for x, y in val_loader:
            pred, ae, eu, kl = net_wrapper.predict(x)
            avg_val_acc += (pred == y).float().mean().item()
        avg_val_acc /= len(val_loader)

        print(
            f"Epoch {epoch + 1:>3} / {EPOCHS:>3} --- "
            f"Training: {avg_train_loss:>10.2f} / {avg_train_acc:.2%} --- "
            f"Validation: {avg_val_acc:.2%}"
        )


if __name__ == "__main__":
    print("Started Training")
    main()
    print("Finished Training")
