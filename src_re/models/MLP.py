import torch
import torch.nn as nn

class CartpoleMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, act_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Output logits; softmax is applied externally for differentiable sampling
        return self.model(x)
    
class ValueMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=[64, 64]):
        super().__init__()
        layers = []
        input_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
