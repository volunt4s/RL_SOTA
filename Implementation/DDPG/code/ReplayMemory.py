import collections
import random
import torch

class ReplayMemory():
    def __init__(self, buffer_size, batch_size):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def put(self, transition):
        self.buffer.append(transition)
    
    def size(self):
        return len(self.buffer)

    def sample(self):
        minibatch = random.sample(self.buffer, self.batch_size)
        state_lst, action_lst, reward_lst, next_state_lst, done_lst = [], [], [], [], []
        
        for transition in minibatch:
            state, action, reward, next_state, done = transition
            state_lst.append(state)
            action_lst.append(action)
            reward_lst.append(reward)
            next_state_lst.append(next_state)
            done_lst.append(done)
        
        return torch.tensor(state_lst, dtype=torch.float), torch.tensor(action_lst, dtype=torch.float32), torch.tensor(reward_lst, dtype=torch.float32), torch.tensor(next_state_lst, dtype=torch.float32), torch.tensor(done_lst, dtype=torch.float32)