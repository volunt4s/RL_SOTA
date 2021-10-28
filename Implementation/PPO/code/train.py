import torch
import torch.nn.functional as F

GAMMA = 0.98
LAMBDA = 0.95
EPS = 0.2

def train(buffer, model, optimizer):
    state_lst, action_lst, reward_lst, obs_lst, done_lst, prob_lst = buffer.make_batch()
    
    for i in range(3):
        td_target = reward_lst + (GAMMA * model.v(obs_lst) * done_lst)
        delta = td_target - model.v(state_lst)
        delta = delta.detach().numpy()

        advantage_lst = []
        advantage = 0.0
        for delta_t in delta[::-1]:
            advantage = GAMMA * LAMBDA * advantage + delta_t[0]
            advantage_lst.append(advantage)
        advantage_lst.reverse()
        advantage = torch.tensor(advantage_lst, dtype=torch.float32)

        pi = model.pi(state_lst, softmax_dim = 1)
        pi_action = pi.gather(1, action_lst)
        ratio = torch.exp(torch.log(pi_action) - torch.log(prob_lst))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1-EPS, 1+EPS) * advantage
        loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(model.v(state_lst), td_target.detach())

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        buffer.clear()
