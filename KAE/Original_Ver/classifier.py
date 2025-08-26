# import torch 
import torch.nn as nn
# import torch.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out = self.fc_layers(x)
        return out # Outputing logits, not probability

class deeper_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(deeper_MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out = self.fc_layers(x)
        return out # Outputing logits, not probability
    
class deepdeeper_MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(deepdeeper_MLP, self).__init__()
        self.fc_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        out = self.fc_layers(x)
        return out # Outputing logits, not probability