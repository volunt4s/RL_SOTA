import collections
import torch
import random

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque()
    
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, batch_size)
        state_lst, action_lst, reward_lst, obs_lst, done_lst = [], [], [], [], []

        for transition in minibatch:
            state, action, reward, obs, done = transition

            state_lst.append(state)
            action_lst.append(action)
            reward_lst.append(reward)
            obs_lst.append(obs)
            done_lst.append(done)

        return torch.tensor(state_lst, dtype=torch.float32), torch.tensor(action_lst), torch.tensor(reward_lst, dtype=torch.float32), torch.tensor(obs_lst), torch.tensor(done_lst)

    def size(self):
        return len(self.buffer)