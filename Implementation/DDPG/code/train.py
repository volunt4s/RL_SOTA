import torch
import torch.nn.functional as F

def train(actor, actor_target, critic, critic_target, memory, gamma, actor_optim, critic_optim):
    state_lst, action_lst, reward_lst, next_state_lst, done_lst = memory.sample()
    action_lst, reward_lst, done_lst = action_lst.unsqueeze(1), reward_lst.unsqueeze(1), done_lst.unsqueeze(1)
    next_action_lst = actor_target(next_state_lst)
    # Critic loss
    td_target = reward_lst + gamma * critic_target(next_state_lst, next_action_lst) * done_lst
    critic_loss = F.smooth_l1_loss(critic(state_lst, action_lst), td_target.detach())
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()
    # Actor loss
    actor_loss = -critic(state_lst, actor(state_lst)).mean()
    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()