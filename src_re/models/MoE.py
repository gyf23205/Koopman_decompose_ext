import torch
import torch.nn as nn

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, num_layers=1):
        super(MoE, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(input_dim, num_experts))
        # layers.append(nn.Softmax(dim=-1))
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.gate(x) 
        return outputs
    

class MoEProb(nn.Module):
    def __init__(self, input_dim, num_experts, num_layers=1):
        super(MoEProb, self).__init__()
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, input_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(input_dim, num_experts))
        layers.append(nn.Softmax(dim=-1))
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.gate(x) 
        return outputs
    
class MoESimple(nn.Module):
    def __init__(self, input_dim, num_experts, num_layers=1):
        super(MoESimple, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, num_experts))
        layers.append(nn.Softmax(dim=-1))
        self.gate = nn.Sequential(*layers)

    def forward(self, x):
        outputs = self.gate(x) 
        return outputs