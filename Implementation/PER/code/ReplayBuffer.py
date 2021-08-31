import collections
import torch
import random
import numpy as np

class ReplayBuffer():
    def __init__(self, buffer_maxlen, epsilon_sample, alpha, beta):
        self.buffer = collections.deque(maxlen=buffer_maxlen)
        self.epsilon_sample = epsilon_sample
        self.alpha = alpha
        self.beta = beta
        
    def put(self, step_data):
        # step_data = [td_error, transition]
        self.buffer.append(step_data)

    def get_priority(self):
        # rank-based prioritization
        td_errors = []
        for i in range(len(self.buffer)):
            td_error_item= self.buffer[i][0].detach().numpy().item()
            td_errors.append(td_error_item)
        priority = np.array(td_errors) + self.epsilon_sample
        priority = priority ** self.alpha
        
        return priority

    def update_priority(self, td_error_lst, selected_idx):
        for i in range(len(td_error_lst)):
            print("BEFORE UPDATE : ", self.buffer[selected_idx[i]][0])
            self.buffer[selected_idx[i]][0] = td_error_lst[i]
            print("AFTER UPDATE : ", self.buffer[selected_idx[i]][0])
   
    def sample(self, batch_size):
        priority = self.get_priority()
        probability = priority / np.sum(priority)
        idx_lst = np.arange(len(self.buffer))
        selected_idx = np.random.choice(idx_lst, size=batch_size, p=probability)
        minibatch = []
        for i in range(batch_size):
            minibatch.append(self.buffer[selected_idx[i]][1])
        
        state_lst, action_lst, reward_lst, obs_lst, done_lst = [], [], [], [], []

        for transition in minibatch:
            state, action, reward, obs, done = transition

            state_lst.append(state)
            action_lst.append(action)
            reward_lst.append(reward)
            obs_lst.append(obs)
            done_lst.append(done)

        return torch.tensor(state_lst, dtype=torch.float32), torch.tensor(action_lst), torch.tensor(reward_lst, dtype=torch.float32), torch.tensor(obs_lst), torch.tensor(done_lst, dtype=torch.float32), selected_idx

    def size(self):
        return len(self.buffer)