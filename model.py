import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description="A DeepLearning model using PyTorch to classify Fashion MNIST images")

parser.add_argument('-d', '--device', type=str, help='The device on which PyTorch should do all the Tensor calculations on, defaults to \'cpu\' if not available', default='cpu')
parser.add_argument('-e', '--epochs', type=int, help='The number of epochs used to train the model', default=20)
parser.add_argument('-l', '--learnrate', type=float, help='The learnrate for the optimizer', default=0.001)
parser.add_argument('-b', '--batchsize', type=int, help='The batchsize for the data loader', default=64)

args = parser.parse_args()

device = torch.device(args.device if torch.backends.mps.is_available() else 'cpu')

print(f"Running on the model on \"{device}\"")

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(6272, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        return self.stack(x)

def train(model, data_loader, optimizer, loss_function, epochs):
    print("Training model...")

    model.train()
    loss_averages = []

    for epoch in range(epochs):
        losses = []
        print(f"Epoch {epoch+1}/{epochs}")
        for i, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(f'\r{(i+1) / len(data_loader) * 100:.2f}%', end='', flush=True)
        loss_averages.append(np.array(losses).mean())
        print(f"\nDone, the final loss is {loss_averages[-1]}")

    plt.plot(loss_averages)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")


def evaluate(model, test_data):
    print("Evaluating model...")

    model.eval()
    correct = 0
    for i, (x, y) in enumerate(test_data):
        x = x.to(device)
        prediction = model(x).argmax(dim=-1).item()
        if prediction == y:
            correct += 1
        print(f'\r{(i + 1) / len(test_data) * 100:.2f}%', end='', flush=True)
    print(f"Done, the model has an accuracy of {evaluation * 100}%")
    return correct / len(test_data)

print("Fetching data...")

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

model = Model().to(device)

train(
    model,
    DataLoader(training_data, args.batchsize, True),
    torch.optim.Adam(model.parameters(), args.learnrate),
    nn.CrossEntropyLoss(),
    epochs=args.epochs
)

evaluation = evaluate(model, test_data)

plt.show()