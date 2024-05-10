import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_drones, n_obs_per_drone, n_actions_per_drone):
        super(DQN, self).__init__()
        self.n_drones = n_drones
        self.n_obs_per_drone = n_obs_per_drone
        self.n_actions_per_drone = n_actions_per_drone
        self.action_values = 3  # Each action can be 0, 1, or 2
        input_dim = n_drones * n_obs_per_drone  # Total input dimension
        output_dim = n_drones * n_actions_per_drone * self.action_values  # Output dimension per action

        # Layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, output_dim)

    def forward(self, x):
        # Flatten input if it's not already flattened
        x = x.view(-1, self.n_drones * self.n_obs_per_drone)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # Reshape output to (n_drones, n_actions_per_drone, action_values)
        x = x.view(-1, self.n_drones, self.n_actions_per_drone, self.action_values)
        # Apply softmax for each action across the possible values
        x = F.softmax(x, dim=-1)  # Softmax across the last dimension (action_values)
        return x


if __name__ == "__main__":
    dqn_model = DQN(n_drones=10, n_obs_per_drone=6, n_actions_per_drone=2)
    sample_input = torch.rand(1, 10, 6)  # Batch size of 1, 10 drones, 6 observations each
    output = dqn_model(sample_input)
    print(output.shape)  # Should print torch.Size([1, 10, 2, 3])
    print(output)  # Outputs are probabilities for each action being 0, 1, or 2

    action_indices = torch.argmax(output, dim=-1)

    # action_indices now contains the indices of the actions with the highest probabilities
    # Its shape will be [batch_size, n_drones, n_actions_per_drone]
    print(action_indices.shape)
    print(action_indices)
    # print(action_indices.reshape(10, 2))
    print(action_indices.squeeze(0))
