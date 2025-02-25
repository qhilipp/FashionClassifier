import model
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import optuna
from optuna.pruners import MedianPruner
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="A DeepLearning model using PyTorch and Optuna to classify Fashion MNIST images")

parser.add_argument('-l', '--load', type=str, help='The name of the file to load the model from', default=None)
parser.add_argument('-s', '--save', type=str, help='The name of the file the model should be saved to', default=None)
parser.add_argument('-d', '--device', type=str, help='The device on which PyTorch should do all the Tensor calculations on, defaults to \'cpu\' if not available', default='cpu')
parser.add_argument('-e', '--epochs', type=int, help='The number of epochs used to train the model', default=20)
parser.add_argument('-t', '--trials', type=int, help='The number of trials that are made to find optimal hyper parameters', default=5)
parser.add_argument('-te', '--trial_epochs', type=int, help='The number of epochs used per trial to train the model when finding optimal hyper parameters', default=2)

args = parser.parse_args()

device = torch.device(args.device if torch.backends.mps.is_available() else 'cpu')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def evaluate_model(model):
    label_map = [
        "T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
    ]

    random_indices = np.random.choice(len(test_data), 15, replace=False)

    subset = [test_data[i] for i in random_indices]

    fig, axes = plt.subplots(3, 5, figsize=(10, 7))

    for i, ax in enumerate(axes.flat):
        ax.imshow(subset[i][0][0], cmap='gray')
        prediction = F.softmax(model(torch.tensor(subset[i][0])))
        predicted_label = prediction.argmax(dim=1).item()
        ax.set_title(
            f'Prediction: {label_map[predicted_label]}\nConfidence: {prediction[0][predicted_label].item() * 100:.2f}%\nActual: {label_map[subset[i][1]]}',
            fontsize=8,
            loc='left'
        )
        ax.axis('off')

    plt.tight_layout()

    accuracy = model.evaluate(test_data, device)
    fig.canvas.manager.set_window_title(f'Accuracy: {accuracy * 100:.2f}%')

    plt.show()

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
    evaluate_model(model)
else:
    print("Finding approximation for optimal hyper parameters...")

    study = optuna.create_study(direction='minimize', pruner=MedianPruner(n_warmup_steps=0))
    study.optimize(lambda trial: model.objective(trial, device, training_data, args.trials, args.trial_epochs), n_trials=args.trials)

    print(f"Done, found {study.best_params} as optimal hyper parameters")

    model = model.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), study.best_params['learn_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    model.fit(
        data_loader=DataLoader(training_data, study.best_params['batch_size'], True),
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=nn.CrossEntropyLoss(),
        epochs=args.epochs,
        device=device,
        verbose=True
    )

    evaluate_model(model)


if args.save is not None:
    torch.save(model.state_dict(), f"models/{args.save}")