from ReplayBuffer import ReplayBuffer
from DDQNAgent import DDQNAgent
import torch.nn.functional as F

def train_ddqn(agent_train, agent_target, buffer, optimizer, batch_size, gamma):
    state_lst, action_lst, reward_lst, obs_lst, done_lst = buffer.sample(batch_size)
    action_lst, reward_lst, done_lst = action_lst.unsqueeze(1), reward_lst.unsqueeze(1), done_lst.unsqueeze(1)
    
    q_lst = agent_train(state_lst).gather(1, action_lst)
    # DDQN
    target_action_lst = agent_train(obs_lst).max(1)[1].unsqueeze(1)
    q_prime_lst = agent_target(obs_lst).gather(1, target_action_lst)
    td_target_lst = reward_lst + gamma * q_prime_lst * done_lst
    
    loss = F.smooth_l1_loss(q_lst, td_target_lst)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def train_dqn(agent_train, agent_target, buffer, optimizer, batch_size, gamma):
    state_lst, action_lst, reward_lst, obs_lst, done_lst = buffer.sample(batch_size)
    action_lst, reward_lst, done_lst = action_lst.unsqueeze(1), reward_lst.unsqueeze(1), done_lst.unsqueeze(1)
    
    q_lst = agent_train(state_lst).gather(1, action_lst)
    q_prime_lst = agent_target(obs_lst).max(1)[0].unsqueeze(1)
    td_target_lst = reward_lst + gamma * q_prime_lst * done_lst

    loss = F.smooth_l1_loss(q_lst, td_target_lst)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()