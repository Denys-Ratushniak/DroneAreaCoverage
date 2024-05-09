import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, input_dim, n_drones):
        super(Actor, self).__init__()
        action_dim = 3  # TODO:denys bad code here
        self.n_drones = n_drones
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_drones * 2 * action_dim),
        )

    def forward(self, state):
        x = self.network(state)
        x = x.view(-1, self.n_drones, 2, 3)  # Reshape to (batch_size, n_drones, 2 actions, 3 possibilities)
        return nn.functional.softmax(x, dim=-1)


class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.network(state)
