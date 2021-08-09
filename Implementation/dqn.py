import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
import matplotlib.pyplot as plt

env = gym.make("LunarLander-v2").unwrapped

class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn>BatchNorm2d(32)


def main():
    env.reset()
    
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    print(screen.shape)

if __name__ == "__main__":
    main()