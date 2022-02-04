import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import soft_update

# Actor class
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # Neural net composition
        self.l1 = nn.Linear(8, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 2)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = torch.tanh(self.l3(x))
        return mu

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # For Q1
        self.l1 = nn.Linear(10, 256) # 2(State dim) + 1(Action dim)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # For Q2
        self.l4 = nn.Linear(10, 256) # 2(State dim) + 1(Action dim)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)    

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    def q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class TD3():
    def __init__(self, gamma, tau, policy_noise, noise_clip):
        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=5e-4)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=5e-4)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.gamma = gamma
        self.tau = tau
        self.step_cnt = 0

    def select_action(self, state):
        action = self.actor(torch.from_numpy(state).float())
        return action

    def train(self, replay_buffer):
        self.step_cnt += 1

        state_lst, action_lst, reward_lst, next_state_lst, done_lst = replay_buffer.sample()
        reward_lst, done_lst = reward_lst.unsqueeze(1), done_lst.unsqueeze(1)

        with torch.no_grad():
            noise = torch.randn_like(action_lst) * self.policy_noise
            noise = np.clip(noise, -self.noise_clip, self.noise_clip)
            
            next_action_lst = self.actor_target(next_state_lst) + noise
            next_action_lst = np.clip(next_action_lst, -1, 1)

            target_Q1, target_Q2 = self.critic_target(next_state_lst, next_action_lst)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_lst + self.gamma * target_Q * done_lst

        current_Q1, current_Q2 = self.critic(state_lst, action_lst)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Every 2 step delayed
        if self.step_cnt % 2 == 0:
            actor_loss = -self.critic.q1(state_lst, self.actor(state_lst)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.critic_target, self.critic, self.tau)
            soft_update(self.actor_target, self.actor, self.tau)


    def save(self):
        torch.save(self.actor.state_dict(), "trained_actor")
