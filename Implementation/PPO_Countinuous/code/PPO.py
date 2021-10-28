import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Env : LunarLanderContinuous-v2 -> Continuous action space
State Space : 8x1 -> Position, Velocity, Angle, Leg contact
Action Space : Continuous [main throttle[-1, 1], side throttle[-1, 1]]

'''

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        # Neural net composition
        self.fc1 = nn.Linear(8, 128)
        self.fc_mu = nn.Linear(128, 2) # Action space size = 2
        self.fc_std = nn.Linear(128, 2)
        self.fc_v = nn.Linear(128, 1) # Value network -> Advantage 구하기 위함
    
    # Get policy mu, std for normal distribution through neural net
    def pi(self, x):
        x = F.relu(self.fc1(x))
        mu = torch.tanh(self.fc_mu(x))
        std = F.softplus(self.fc_std(x))
        return mu, std
    
    # Get value v through neural net
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    