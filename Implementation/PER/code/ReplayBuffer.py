import collections
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

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
        # get priority(td error + epsilon) in whole memory
        td_errors = []
        for i in range(len(self.buffer)):
            td_error_item= self.buffer[i][0].detach().numpy().item()
            td_errors.append(td_error_item)
        priority = np.array(td_errors) + self.epsilon_sample
        priority = priority ** self.alpha
        
        return priority

    def update_priority(self, td_error_lst, selected_idx):
        for i in range(len(td_error_lst)):
            self.buffer[selected_idx[i]][0] = td_error_lst[i]

    def sample(self, batch_size):
        priority = self.get_priority()
        probability = priority / np.sum(priority)
        idx_lst = np.arange(len(self.buffer))
        selected_idx = np.random.choice(idx_lst, size=batch_size, p=probability)
        selected_td_error = []
        minibatch = []

        for i in range(batch_size):
            minibatch.append(self.buffer[selected_idx[i]][1])
            selected_td_error.append(self.buffer[selected_idx[i]][0].detach().numpy().item())

        plt.plot(np.arange(32), selected_td_error)
        plt.ylim((0, 150))
        plt.savefig('tderror.png')

        state_lst, action_lst, reward_lst, obs_lst, done_lst = [], [], [], [], []

        for transition in minibatch:
            state, action, reward, obs, done = transition

            state_lst.append(state)
            action_lst.append(action)
            reward_lst.append(reward)
            obs_lst.append(obs)
            done_lst.append(done)

        return torch.tensor(state_lst, dtype=torch.float32), torch.tensor(action_lst), torch.tensor(reward_lst, dtype=torch.float32), torch.tensor(obs_lst), torch.tensor(done_lst, dtype=torch.float32), selected_idx
    
    def get_is_weight(self, selected_idx, batch_size):
        # get IS-weight for sampled element
        priority = self.get_priority()
        probability = priority / np.sum(priority)
        # is_weight -> IS-weight for all element
        is_weight = (self.size() * probability) ** (-self.beta)
        selected_is_weight = []
        for i in range(batch_size):
            selected_is_weight.append(is_weight[selected_idx[i]])
        selected_is_weight = selected_is_weight / np.max(is_weight)
        
        return selected_is_weight

    def improving_beta(self, beta_end, beta_rate):
        if self.beta < beta_end:
            self.beta *= beta_rate

    def size(self):
        return len(self.buffer)

    def get_beta(self):
        return self.beta