import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Env : MountainCarContinuous-v0
State space : [Position(-1.2~0.6), Velocity(-0.07, 0.07)]
Action space : [Power(-1~1)] -> Countinuous control
'''

# Actor class
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # Neural net composition
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.fc_mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # Neural net composition
        self.fc_s = nn.Linear(2, 64)
        self.fc_a = nn.Linear(1, 64)
        self.fc_q = nn.Linear(128, 32)
        self.fc_out = nn.Linear(32, 1)    

    def forward(self, x, a):
        s = F.relu(self.fc_s(x))
        a = F.relu(self.fc_a(a))
        concat = torch.cat([s, a], dim=1)
        q = F.relu(self.fc_q(concat))
        q = self.fc_out(q)
        return q