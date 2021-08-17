import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

'''
Env : LunarLander-v2 -> Non continuous
State Space : 8x1 -> Position, Velocity, Angle, Leg contact
Action Space : Discrete(4) -> [Non, Left, Main, Right]
'''

class DDQNAgent(nn.Module):
    def __init__(self):
        super(DDQNAgent, self).__init__()
        # neural net composition
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, state, epsilon):
        out = self.forward(state)
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            # argmax() -> torch return, item() -> unwrap torch
            return out.argmax().item()

    def numpy_to_torch(self, element):
        return torch.from_numpy(element) 