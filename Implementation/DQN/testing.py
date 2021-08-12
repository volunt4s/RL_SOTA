import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # Conv Layer 정의
        self.conv1 = nn.Conv2d(3, 6, kernel_size=7, stride=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=4, stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(432, 216)
        self.fc2 = nn.Linear(216, 12)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x
    
    def sample_action(self, state, epsilon):
        out = self.forward(state)
        if random.random() < epsilon:
            return random.randint(0, 11)
        else:
            return out.argmax().item()


def main():
    env = gym.make("CarRacing-v0")
    test_agent = DQN()
    test_agent.load_state_dict(torch.load("trained_model"))

    action_space = np.array([
        (-1, 0.6, 0.2), (0, 0.6, 0.2), (1, 0.6, 0.2), # Slow go
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Brake
        (-1, 0.7, 0), (0, 0.7, 0), (1, 0.7, 0),       # Fast go
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)        # no accel
    ])

    for epi in range(100):
        done = False
        state = env.reset()
        reward_sum = 0
        while not done:
            env.render()
            reward_temp = 0
            torch_state = torch.from_numpy(state.copy()).unsqueeze(0).float()
            torch_state = torch_state.permute(0, 3, 1, 2) / 255.0
            action_space_idx = test_agent.sample_action(torch_state, 0)
            action = action_space[action_space_idx]
            next_state, reward, done, _ = env.step(action)
            reward_temp += reward
            cnt = cnt + 1 if reward_temp < 0 else 0

            if cnt > 35:
                break
            
            state = next_state


if __name__ == "__main__":
    main()