import numpy as np
import torch
import matplotlib.pyplot as plt
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


    def fit(self, data_loader, optimizer, loss_function, epochs, device, verbose=False) -> float:
        if verbose:
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
            print(f"\nDone, the final loss is {loss_averages[-1]}")

        return loss_averages[-1]


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


def objective(trial, device, training_data, trials, trial_epochs) -> float:
    learn_rate = trial.suggest_float("learn_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 32, 128, log=True)

    model = Model().to(device)

    print(f"Trial {trial.number+1}/{trials}")

    loss = model.fit(
        data_loader=DataLoader(training_data, batch_size, True),
        optimizer=torch.optim.Adam(model.parameters(), learn_rate),
        loss_function=nn.CrossEntropyLoss(),
        epochs=trial_epochs,
        device=device
    )

    print(f"Done, arguments {trial.params} resulted in {loss}")

    return loss
