import torch.nn as nn
import torch.nn.functional as F

'''
Env : LunarLander-v2 -> Non continuous
State Space : 8x1 -> Position, Velocity, Angle, Leg contact
Action Space : Discrete(4) -> [Non, Left, Main, Right]
'''

class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        # Neural net composition
        self.fc1 = nn.Linear(8, 128)
        self.fc_pi = nn.Linear(128, 4) # Action space size = 4
        self.fc_v = nn.Linear(128, 1) # Value network -> Advantage 구하기 위함
    
    # Get policy pi through neural net
    def pi(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    # Get value v through neural net
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v
    