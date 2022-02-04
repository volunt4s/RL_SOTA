import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import soft_update
from torch.distributions import Normal
'''
BipedalHardcore -> state 24x1 action 4x1
LunarLanderContinuous-v2 -> state 8x1 action 4x1
'''
# Actor class
class Actor(nn.Module):
    def __init__(self, lr_actor, init_alpha, lr_alpha):
        super(Actor, self).__init__()
        # Neural net composition
        self.l1 = nn.Linear(8, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3_1 = nn.Linear(128, 4)
        self.l3_2 = nn.Linear(128, 4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_actor)
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self.l3_1(x)
        std = F.softplus(self.l3_2(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action) # Clip -1 ~ 1 due to Lunarlander
        log_prob -= torch.log(1-torch.tanh(action).pow(2)+1e-7)
        log_prob = log_prob.sum()
        return real_action, log_prob

    def train_actor(self, q1_net, q2_net, mini_batch):
        state_lst, _, _, _, _ = mini_batch
        action_lst, log_prob_lst = self.forward(state_lst)
        q1, q2 = q1_net(state_lst, action_lst), q2_net(state_lst, action_lst)
        min_q = torch.min(q1, q2)
        entropy = -self.log_alpha.exp() * log_prob_lst

        loss = -(min_q + entropy) # For gradient ascent
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob_lst - 4).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


class Critic(nn.Module):
    def __init__(self, lr_critic):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(12, 256) # 8(State dim) + 4(Action dim)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_critic)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
    def train_critic(self, mini_batch, target):
        state_lst, action_lst, reward_lst, _, done_lst = mini_batch
        reward_lst, done_lst = reward_lst.unsqueeze(1), done_lst.unsqueeze(1)
        loss = F.smooth_l1_loss(self.forward(state_lst, action_lst), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class SAC():
    def __init__(self, gamma, tau, lr_actor, lr_critic, init_alpha, lr_alpha):
        self.pi_net = Actor(lr_actor, init_alpha, lr_alpha)
        self.q1_net, self.q2_net, self.q1_target_net, self.q2_target_net = Critic(lr_critic), Critic(lr_critic), Critic(lr_critic), Critic(lr_critic) 
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        action, _ = self.pi_net(torch.from_numpy(state).float())
        return action

    def calc_target(self, pi_net, q1_net, q2_net, mini_batch):
        _, _, reward_lst, next_state_lst, done_lst = mini_batch
        reward_lst, done_lst = reward_lst.unsqueeze(1), done_lst.unsqueeze(1)
        with torch.no_grad():
            next_action_lst, log_prob = pi_net(next_state_lst)
            entropy = -(pi_net.log_alpha.exp() * log_prob)
            q1, q2 = q1_net(next_state_lst, next_action_lst), q2_net(next_state_lst, next_action_lst)
            min_q = torch.min(q1, q2)
            target = reward_lst + self.gamma * done_lst * (min_q + entropy)
        return target

    def train(self, replay_buffer):
        mini_batch = replay_buffer.sample()
        td_target = self.calc_target(self.pi_net, self.q1_target_net, self.q2_target_net, mini_batch)
        self.q1_net.train_critic(mini_batch, td_target)
        self.q2_net.train_critic(mini_batch, td_target)
        self.pi_net.train_actor(self.q1_net, self.q2_net, mini_batch)
        soft_update(self.q1_target_net, self.q1_net, self.tau)
        soft_update(self.q2_target_net, self.q2_net, self.tau)

    def save(self):
        torch.save(self.pi_net.state_dict(), "trained_actor")
    
    def get_alpha(self):
        return self.pi_net.log_alpha.exp()
    

