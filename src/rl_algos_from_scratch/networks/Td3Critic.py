import torch.nn.functional as F
from torch import nn


class Td3Critic(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])
        self.hidden_layers = nn.ModuleList()

        self.activation_fn = F.relu

        for i in range(1, len(hidden_dim) - 1):
            layer = nn.Linear(hidden_dim[i], hidden_dim[i + 1])
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_dim[-1], output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation_fn(x)

        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation_fn(x)
        x = self.output_layer(x)

        return x
