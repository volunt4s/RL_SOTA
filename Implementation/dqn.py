from PIL.Image import init
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
MEMORY_SIZE = 1000000
MINIBATCH_SIZE = 32

'''
Env : CarRacing-v0
State : 96 x 96 x 3 pixel -> 전체 지도 Top view
Action : [[-1 ~ +1], [0 ~ +1], [0 ~ +1]] -> 스티어링, 악셀, 브레이크
'''

env = gym.make("CarRacing-v0").unwrapped


class DQN(nn.Module):

    def __init__(self, height, width, outputs):
        super(DQN, self).__init__()
        
        # Conv Layer 정의
        # conv1 -> conv2 -> conv3 -> fc1 -> output
        pass

        # convolution size 공식
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return ((size - (kernel_size - 1) - 1) // stride) + 1


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=MEMORY_SIZE)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, size):
        minibatch = random.sample(self.buffer, MINIBATCH_SIZE)
        state_lst, action_lst, reward_lst, next_state_lst, is_done_lst = [], [], [], [], []

        for transition in minibatch:
            state, action, reward, next_state, is_done = transition
            state_lst.append(state)
            action_lst.append(action)
            reward_lst.append(reward)
            next_state_lst.append(next_state)
            is_done_lst.append(is_done)
        
        return torch.tensor(state_lst), torch.tensor(action_lst), torch.tensor(reward_lst), torch.tensor(next_state_lst), torch.tensor(is_done_lst)
            
    def size(self):
        return len(self.buffer)

def main():
    pass

if __name__ == "__main__":
    main()