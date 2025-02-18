import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import argparse
import optuna

parser = argparse.ArgumentParser(description="A DeepLearning model using PyTorch and Optuna to classify Fashion MNIST images")

parser.add_argument('-l', '--load', type=str, help='The name of the file to load the model from', default=None)
parser.add_argument('-s', '--save', type=str, help='The name of the file the model should be saved to', default=None)
parser.add_argument('-d', '--device', type=str, help='The device on which PyTorch should do all the Tensor calculations on, defaults to \'cpu\' if not available', default='cpu')
parser.add_argument('-e', '--epochs', type=int, help='The number of epochs used to train the model', default=20)
parser.add_argument('-t', '--trials', type=int, help='The number of trials that are made to find optimal hyper parameters', default=5)
parser.add_argument('-te', '--trial_epochs', type=int, help='The number of epochs used per trial to train the model when finding optimal hyper parameters', default=2)

args = parser.parse_args()
plt.ion()

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

def train(model, data_loader, optimizer, loss_function, epochs, verbose=False) -> float:
    if verbose:
        print("Training model...")

    model.train()
    loss_averages = []

    for epoch in range(epochs):
        losses = []
        if verbose:
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
            if verbose:
                print(f'\r{(i+1) / len(data_loader) * 100:.2f}%', end='', flush=True)

        loss_averages.append(np.array(losses).mean())
        if verbose:
            print()

    if verbose:
        plt.plot(loss_averages)
        plt.title("Training loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

    if verbose:
        print(f"\nDone, the final loss is a loss of {loss_averages[-1]}")

    return loss_averages[-1]


def evaluate(model, test_data) -> float:
    print("Evaluating model...")

    model.eval()
    correct = 0
    for i, (x, y) in enumerate(test_data):
        x = x.to(device)
        prediction = model(x).argmax(dim=-1).item()
        if prediction == y:
            correct += 1
        print(f'\r{(i + 1) / len(test_data) * 100:.2f}%', end='', flush=True)

    evaluation = correct / len(test_data)
    print(f"\nDone, the model has an accuracy of {evaluation * 100}%")

    return evaluation

def objective(trial):
    learn_rate = trial.suggest_float("learn_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    model = Model().to(device)

    print(f"Trial {trial.number+1}/{args.trials}")

    loss = train(
        model,
        DataLoader(training_data, batch_size, True),
        torch.optim.Adam(model.parameters(), learn_rate),
        nn.CrossEntropyLoss(),
        epochs=args.trial_epochs
    )

    print(f"Done, arguments {trial.params} resulted in {loss}")

    return loss

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

if args.load is not None:
    model = torch.load(f"models/{args.load}", weights_only=False)
else:
    print("Finding approximation for optimal hyper parameters...")

    study = optuna.create_study(direction='minimize') #, pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=args.trials)

    print(f"Done, found {study.best_params} as optimal hyper parameters")

    model = Model().to(device)

    train(
        model,
        DataLoader(training_data, study.best_params['batch_size'], True),
        torch.optim.Adam(model.parameters(), study.best_params['learn_rate']),
        nn.CrossEntropyLoss(),
        epochs=args.epochs,
        verbose=True
    )

    plt.show()

evaluate(model, test_data)

if args.save is not None:
    torch.save(model, f"models/{args.save}")