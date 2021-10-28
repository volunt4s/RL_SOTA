import torch

class Buffer():
    def __init__(self):
        self.buffer = []
    
    def put(self, transition):
        # Put each transition to buffer
        self.buffer.append(transition)
    
    def make_batch(self):
        # Make batch data from buffer, Return it to tensor
        state_lst, action_lst, reward_lst, obs_lst, done_lst, prob_lst = [], [], [], [], [], []
        for transition in self.buffer:
            state, action, reward, obs, done, prob = transition
            state_lst.append(state)
            action_lst.append([action])
            reward_lst.append([reward])
            obs_lst.append(obs)
            done_lst.append([done])
            prob_lst.append([prob])

        return torch.tensor(state_lst, dtype=torch.float32), torch.tensor(action_lst), torch.tensor(reward_lst, dtype=torch.float32), torch.tensor(obs_lst, dtype=torch.float32), torch.tensor(done_lst, dtype=torch.float32), torch.tensor(prob_lst, dtype=torch.float32)
    
    def size(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()