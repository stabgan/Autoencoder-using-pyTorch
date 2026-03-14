# Stacked Autoencoder for Movie Recommendations
# PyTorch implementation trained on MovieLens 100K

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

# ---------------------------------------------------------------------------
# Resolve paths relative to this script so it works from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load and prepare the MovieLens 100K training and test sets."""
    training_set = pd.read_csv(
        os.path.join(SCRIPT_DIR, "ml-100k", "u1.base"), delimiter="\t"
    )
    training_set = np.array(training_set, dtype=int)

    test_set = pd.read_csv(
        os.path.join(SCRIPT_DIR, "ml-100k", "u1.test"), delimiter="\t"
    )
    test_set = np.array(test_set, dtype=int)

    nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
    nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

    return training_set, test_set, nb_users, nb_movies


def convert(data, nb_users, nb_movies):
    """Convert raw ratings into a user×movie matrix (zeros for unrated)."""
    new_data = []
    for id_user in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SAE(nn.Module):
    """Stacked Autoencoder with sigmoid activations and linear output."""

    def __init__(self, nb_movies):
        super().__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(sae, criterion, optimizer, training_set, nb_users, nb_movies, device, nb_epoch=200):
    """Train the autoencoder and print per-epoch loss."""
    sae.train()
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0.0
        s = 0.0
        for id_user in range(nb_users):
            input_data = training_set[id_user].unsqueeze(0).to(device)
            target = input_data.clone().detach()
            if torch.sum(target > 0) > 0:
                output = sae(input_data)
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_movies / float(torch.sum(target > 0) + 1e-10)
                optimizer.zero_grad()
                loss.backward()
                train_loss += np.sqrt(loss.item() * mean_corrector)
                s += 1.0
                optimizer.step()
        print(f"epoch: {epoch}  loss: {train_loss / s:.4f}")


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
def test(sae, criterion, training_set, test_set, nb_users, nb_movies, device):
    """Evaluate the autoencoder on the held-out test set."""
    sae.eval()
    test_loss = 0.0
    s = 0.0
    with torch.no_grad():
        for id_user in range(nb_users):
            input_data = training_set[id_user].unsqueeze(0).to(device)
            target = test_set[id_user].unsqueeze(0).to(device)
            if torch.sum(target > 0) > 0:
                output = sae(input_data)
                output[target == 0] = 0
                loss = criterion(output, target)
                mean_corrector = nb_movies / float(torch.sum(target > 0) + 1e-10)
                test_loss += np.sqrt(loss.item() * mean_corrector)
                s += 1.0
    print(f"test loss: {test_loss / s:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Device selection: use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    training_raw, test_raw, nb_users, nb_movies = load_data()

    # Build user×movie matrices and convert to tensors
    training_set = torch.FloatTensor(convert(training_raw, nb_users, nb_movies))
    test_set = torch.FloatTensor(convert(test_raw, nb_users, nb_movies))

    # Model, loss, optimizer
    sae = SAE(nb_movies).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

    # Train
    train(sae, criterion, optimizer, training_set, nb_users, nb_movies, device, nb_epoch=200)

    # Test
    test(sae, criterion, training_set, test_set, nb_users, nb_movies, device)

    # Save trained weights
    save_path = os.path.join(SCRIPT_DIR, "sae_weights.pth")
    torch.save(sae.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


if __name__ == "__main__":
    main()
