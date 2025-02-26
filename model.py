from typing import Any

import numpy as np
import optuna
import torch
import matplotlib.pyplot as plt
from optuna import TrialPruned
from optuna.pruners import MedianPruner
from torch.utils.data import DataLoader
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 256),
            nn.Dropout(0.3),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, 28, 28)
        return self.stack(x)


    def fit(self, data_loader, optimizer, scheduler, loss_function, epochs, device, trial=None, loss_map=None, verbose=0):
        if verbose > 0:
            print("Training model...")

        self.train()
        loss_averages = []

        for epoch in range(epochs):
            losses = []
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}")

            for i, (x, y) in enumerate(data_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                prediction = self(x)
                loss = loss_function(prediction, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                if verbose > 0:
                    print(f'\r{(i+1) / len(data_loader) * 100:.2f}%', end='', flush=True)

            loss_averages.append(np.array(losses).mean())

            scheduler.step(loss_averages[-1])

            if verbose > 0:
                print()

            if trial is not None:
                trial.report(loss_averages[-1], epoch)

                if trial.should_prune():
                    if verbose > 0:
                        print(f"Trial {trial.number+1} was pruned!")
                        loss_map.append(loss_averages + [np.nan] * (epochs - len(loss_averages)))
                    raise TrialPruned()

        if verbose > 1:
            plt.plot(loss_averages)
            plt.title("Training loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")

        if verbose > 0:
            print(f"\nDone, the final loss is {loss_averages[-1]}")

        if loss_map is not None:
            loss_map.append(loss_averages)


    def evaluate(self, test_data, device) -> float:
        print("Evaluating model...")

        self.eval()
        correct = 0
        for i, (x, y) in enumerate(test_data):
            x = x.to(device)
            prediction = self(x).argmax(dim=-1).item()
            if prediction == y:
                correct += 1
            print(f'\r{(i + 1) / len(test_data) * 100:.2f}%', end='', flush=True)

        evaluation = correct / len(test_data)
        print(f"\nDone, the model has an accuracy of {evaluation * 100}%")

        return evaluation


def tune_hyperparameters(device, training_data, trials, trial_epochs, loss_map) -> (dict[str, Any]):
    study = optuna.create_study(direction='minimize', pruner=MedianPruner(n_warmup_steps=0, n_startup_trials=1))

    study.optimize(lambda trial: objective(trial, device, training_data, trials, trial_epochs, loss_map), n_trials=trials)

    return study.best_params


def objective(trial, device, training_data, trials, trial_epochs, loss_map) -> float:
    learn_rate = trial.suggest_float("learn_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [24, 32, 48])

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), learn_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    print(f"Trial {trial.number+1}/{trials}")

    model.fit(
        data_loader=DataLoader(training_data, batch_size, True),
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=nn.CrossEntropyLoss(),
        epochs=trial_epochs,
        trial=trial,
        verbose=1,
        loss_map=loss_map,
        device=device
    )

    losses = loss_map[-1]

    print(f"Done, arguments {trial.params} resulted in a loss of {losses[-1]}")

    return losses[-1]
