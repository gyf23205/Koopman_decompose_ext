import torch
import torch.nn as nn

class MoE(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        outputs = self.gate(x) 
        return outputs