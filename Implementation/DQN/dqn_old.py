import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import collections
import random

# HYPER PARAMETERS
BUFFER_SIZE = 10000
MINIBATCH_SIZE = 32
EPISODES = 1000
EPS_START = 1
EPS_END = 0.01
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.001

'''
Env : CarRacing-v0
State : 96 x 96 x 3 pixel -> 전체 지도 Top view
Action : [[-1 ~ +1], [0 ~ +1], [0 ~ +1]] -> 스티어링, 악셀, 브레이크
'''

env = gym.make("CarRacing-v0")


class DQN(nn.Module):
    def __init__(self, height, width, outputs):
        super(DQN, self).__init__()
        
        # Conv Layer 정의
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)        

        # convolution size 공식
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return ((size - (kernel_size - 1) - 1) // stride) + 1
        
        conv_height = conv2d_size_out(conv2d_size_out(height))
        conv_width = conv2d_size_out(conv2d_size_out(width))
        linear_input_size = conv_height * conv_width * 32
        self.fc1 = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.fc1(x.view(x.size(0), -1))
    
    def sample_action(self, state, epsilon):
        out = self.forward(state)
        if random.random() < epsilon:
            return random.randint(0, 11)
        else:
            return out.argmax().item()


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_SIZE)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, size):
        minibatch = random.sample(self.buffer, size)
        state_lst, action_lst, reward_lst, next_state_lst, done_lst = [], [], [], [], []

        for transition in minibatch:
            state, action, reward, next_state, done_mask = transition
            state_lst.append(state)
            action_lst.append(action)
            reward_lst.append(reward)
            next_state_lst.append(next_state)
            done_lst.append(done_mask)
        
        return torch.tensor(state_lst, dtype=torch.float32), torch.tensor(action_lst), torch.tensor(reward_lst), torch.tensor(next_state_lst, dtype=torch.float32), torch.tensor(done_lst)
            
    def size(self):
        return len(self.buffer)


def train(q_train_net, q_target_net, buffer, optimizer):
    for i in range(10):
        state_lst, action_lst, reward_lst, next_state_lst, done_lst = buffer.sample(MINIBATCH_SIZE)
        
        # RGB -> 0 ~ 256 이므로 / 255
        state_lst = state_lst.permute(0, 3, 1, 2) / 255
        next_state_lst = next_state_lst.permute(0, 3, 1, 2) / 255
        action_lst, reward_lst, done_lst = action_lst.unsqueeze(1), reward_lst.unsqueeze(1), done_lst.unsqueeze(1)
        q_value_lst = q_train_net(state_lst).gather(1, action_lst)
        # max -> return : tensor value, indice -> 값을 가져오기 위해 [0]
        q_prime_value_lst = q_target_net(next_state_lst).max(1)[0].unsqueeze(1)
        target = reward_lst + DISCOUNT_FACTOR * q_prime_value_lst * done_lst
        
        loss = F.smooth_l1_loss(target, q_value_lst)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    q_train_net = DQN(96, 96, 12)
    q_target_net = DQN(96, 96, 12)
    q_target_net.load_state_dict(q_train_net.state_dict())
    buffer = ReplayBuffer()
    optimizer = optim.Adam(q_train_net.parameters(), lr=LEARNING_RATE)

    # Continuous action space -> DQN 적합 x -> 임의로 지정
    action_space = np.array([
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), # Slow go
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Brake
        (-1, 1, 0), (0, 1, 0), (1, 1, 0),       # Fast go
        (-1, 0, 0), (0, 0, 0), (1, 0, 0)        # no accel
    ])
    
    print("=== TRAIN START ===")

    for epi in range(EPISODES):
        #TODO : exponetially decay
        epsilon = max(EPS_END, EPS_START- EPS_END * epi)
        state = env.reset()
        done = False
        reward_sum = 0
        step_cnt = 0
        # During episode
        while not done:
            torch_state = torch.from_numpy(state.copy()).unsqueeze(0).float()
            torch_state = torch_state.permute(0, 3, 1, 2) / 255
            action_space_idx = q_train_net.sample_action(torch_state, epsilon)
            action = action_space[action_space_idx]
            next_state, reward, done, _ = env.step(action)

            # target 계산 위해 false일 떄 0.0, true일 때 1.0인 마스크 선언
            done_mask = 0.0 if done else 1.0
            # 버퍼 데이터 입력
            buffer.put([state, action_space_idx, reward, next_state, done_mask])
            state = next_state
            reward_sum += reward

        if buffer.size() > 5000:
            train(q_train_net, q_target_net, buffer, optimizer)

        if epi % 10 == 0:
            print("updated")
            q_target_net.load_state_dict(q_train_net.state_dict())

        print("EPISODE : {:d} REWARD : {:.1f} EPSILON : {:.1f} BUFFER : {:d}\n".format(epi, reward_sum, epsilon, buffer.size()))

    env.close()


if __name__ == "__main__":
    main()