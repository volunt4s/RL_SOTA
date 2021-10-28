import torch
import torch.nn.functional as F
from torch.distributions import Normal

GAMMA = 0.98
LAMBDA = 0.95
EPS = 0.2
EPOCH = 5

def train(buffer, model, optimizer):
    state_lst, action_lst, reward_lst, obs_lst, done_lst, old_log_prob_lst = buffer.make_batch()
    
    for epoch in range(EPOCH):
        td_target = reward_lst + (GAMMA * model.v(obs_lst) * done_lst)
        delta = td_target - model.v(state_lst)
        delta = delta.detach().numpy()
        
        # Get Advantage
        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = GAMMA * LAMBDA * advantage + delta_t[0]
            advantage_lst.append(advantage)
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float32)

        # Get ratio
        mu, std = model.pi(state_lst)
        dist = Normal(mu, std)
        new_log_prob_lst = dist.log_prob(action_lst)
        ratio = torch.exp(new_log_prob_lst - old_log_prob_lst)
        ratio = ratio.mean()
        # Get loss
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-EPS, 1+EPS) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.v(state_lst), td_target.detach())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

    buffer.clear()
