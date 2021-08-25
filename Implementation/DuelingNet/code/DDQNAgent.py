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
        self.fc_value = nn.Linear(128, 128)
        self.fc_adv = nn.Linear(128, 128)
        self.value = nn.Linear(128, 1)
        self.adv = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))
        value = self.value(value)
        adv = self.adv(adv)

        adv_avg = torch.mean(adv)
        q = value + adv - adv_avg
        return q

    def sample_action(self, state, epsilon):
        out = self.forward(state)
        if random.random() < epsilon:
            return random.randint(0, 3)
        else:
            # argmax() -> torch return, item() -> unwrap torch
            return out.argmax().item()

    def numpy_to_torch(self, element):
        return torch.from_numpy(element)

    def out_qvalue(self, state):
        out = self.forward(state)
        return out