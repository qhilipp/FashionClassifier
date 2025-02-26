import model as mdl
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import optuna
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json

parser = argparse.ArgumentParser(description="A DeepLearning model using PyTorch and Optuna to classify Fashion MNIST images")

parser.add_argument('-l', '--load', type=str, help='The name of the file to load the model from', default=None)
parser.add_argument('-s', '--save', type=str, help='The name of the file the model should be saved to', default=None)
parser.add_argument('-d', '--device', type=str, help='The device on which PyTorch should do all the Tensor calculations on, defaults to \'cpu\' if not available', default='cpu')
parser.add_argument('-e', '--epochs', type=int, help='The number of epochs used to train the model', default=20)
parser.add_argument('-t', '--trials', type=int, help='The number of trials that are made to find optimal hyper parameters', default=5)
parser.add_argument('-te', '--trial_epochs', type=int, help='The number of epochs used per trial to train the model when finding optimal hyper parameters', default=2)
parser.add_argument('-lh', '--load_hyper_parameters', type=str, help='The name of a json file the model should load its hyper parameters from', default=None)
parser.add_argument('-sh', '--save_hyper_parameters', type=str, help='The name of a json file the model should save its hyper parameters to', default=None)

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
        ax.imshow(subset[i][0][0])
        prediction = F.softmax(model(torch.tensor(subset[i][0]).to(device)))
        predicted_label = prediction.argmax(dim=1).item()
        color = 'red' if predicted_label != subset[i][1] else 'black'
        ax.set_title(
            f'Prediction: {label_map[predicted_label]}\nConfidence: {prediction[0][predicted_label].item() * 100:.2f}%\nActual: {label_map[subset[i][1]]}',
            fontsize=8,
            loc='left',
            color=color
        )
        ax.axis('off')

    plt.tight_layout()

    accuracy = model.evaluate(test_data, device)
    fig.canvas.manager.set_window_title(f'Accuracy: {accuracy * 100:.2f}%')

    plt.show()

def train_model():
    if args.load_hyper_parameters is None:
        print("Finding approximation for optimal hyper parameters...")

        loss_map = []
        best_params = mdl.tune_hyperparameters(device, training_data, args.trials, args.trial_epochs, loss_map)

        if args.save_hyper_parameters is not None:
            with open(args.save_hyper_parameters, 'w') as file:
                json.dump(best_params, file, indent=4)

        plt.imshow(loss_map, aspect='auto', cmap='viridis')
        plt.colorbar(label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Trial')
        plt.title('Loss for different hyper parameters')
        plt.show()

        print(f"Done, found {best_params} as optimal hyper parameters")
    else:
        with open(args.load_hyper_parameters, 'r') as file:
            best_params = json.load(file)

            print(f"Loaded these hyper parameters {best_params}")

    model = mdl.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), best_params['learn_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    model.fit(
        data_loader=DataLoader(training_data, best_params['batch_size'], True),
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=nn.CrossEntropyLoss(),
        epochs=args.epochs,
        device=device,
        verbose=2
    )

    evaluate_model(model)

    return model

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
    model = mdl.Model().to(device)
    model.load_state_dict(torch.load(f"models/{args.load}"))
    evaluate_model(model)
else:
    model = train_model()

    if args.save is not None:
        torch.save(model.state_dict(), f"models/{args.save}")