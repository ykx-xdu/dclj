import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.residual = True

    def forward(self, state):
        x = F.leaky_relu(self.ln1(self.fc1(state)), 0.01)
        x = self.dropout(x)
        res1 = x
        x = F.leaky_relu(self.ln2(self.fc2(x)), 0.01)
        x = self.dropout(x)
        if self.residual:
            x = x + res1
        res2 = x
        x = F.leaky_relu(self.ln3(self.fc3(x)), 0.01)
        x = self.dropout(x)
        if self.residual:
            x = x + res2
        action = torch.tanh(self.fc4(x))
        return action
