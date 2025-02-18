import model
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import optuna
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description="A DeepLearning model using PyTorch and Optuna to classify Fashion MNIST images")

parser.add_argument('-l', '--load', type=str, help='The name of the file to load the model from', default=None)
parser.add_argument('-s', '--save', type=str, help='The name of the file the model should be saved to', default=None)
parser.add_argument('-d', '--device', type=str, help='The device on which PyTorch should do all the Tensor calculations on, defaults to \'cpu\' if not available', default='cpu')
parser.add_argument('-e', '--epochs', type=int, help='The number of epochs used to train the model', default=20)
parser.add_argument('-t', '--trials', type=int, help='The number of trials that are made to find optimal hyper parameters', default=5)
parser.add_argument('-te', '--trial_epochs', type=int, help='The number of epochs used per trial to train the model when finding optimal hyper parameters', default=2)

args = parser.parse_args()

device = torch.device(args.device if torch.backends.mps.is_available() else 'cpu')

print(f"Running on the model on \"{device}\"")

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
    model = model.Model().to(device)
    model.load_state_dict(torch.load(f"models/{args.load}"))
else:
    print("Finding approximation for optimal hyper parameters...")

    study = optuna.create_study(direction='minimize') #, pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: model.objective(trial, device, training_data, args.trials, args.trial_epochs), n_trials=args.trials)

    print(f"Done, found {study.best_params} as optimal hyper parameters")

    model = model.Model().to(device)

    model.fit(
        data_loader=DataLoader(training_data, study.best_params['batch_size'], True),
        optimizer=torch.optim.Adam(model.parameters(), study.best_params['learn_rate']),
        loss_function=nn.CrossEntropyLoss(),
        epochs=args.epochs,
        device=device,
        verbose=True
    )

    model.evaluate(test_data, device=device)

    plt.show()


if args.save is not None:
    torch.save(model.state_dict(), f"models/{args.save}")