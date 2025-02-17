import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn as nn

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

def train(model, data_loader, optimizer, loss_function, epochs=20):
    model.train()
    loss_averages = []

    for epoch in range(epochs):
        losses = []
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = loss_function(prediction, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_averages.append(np.array(losses).mean())
        print(f"{(epoch / epochs) * 100}%")

    plt.plot(loss_averages)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

def evaluate(model, test_data):
    model.eval()
    correct = 0
    for x, y in test_data:
        x = x.to(device)
        prediction = model(x).argmax(dim=-1).item()
        if prediction == y:
            correct += 1
    return correct / len(test_data)

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
    DataLoader(training_data, 64, True),
    torch.optim.Adam(model.parameters(), 0.001),
    nn.CrossEntropyLoss(),
    epochs=25
)

print(evaluate(model, test_data))
