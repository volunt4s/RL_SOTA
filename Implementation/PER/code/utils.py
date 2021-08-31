from DDQNAgent import DDQNAgent
import matplotlib.pyplot as plt
import torch

def decaying_eps(epsilon, epsilon_end, epsilon_decay):
    if epsilon > epsilon_end:
        epsilon *= epsilon_decay
        return epsilon
    else:
        return epsilon

def load_network(agent_train, agent_target):
    agent_target.load_state_dict(agent_train.state_dict())

def plot_reward(episode_lst, reward_lst):
    plt.plot(episode_lst, reward_lst)
    plt.xlabel('EPISODES')
    plt.ylabel('SCORE')
    plt.ylim((-400, 300))
    plt.savefig('graph_reward.png')
    plt.close()

def plot_qvalue(episode_lst, qvalue_lst):
    plt.plot(episode_lst, qvalue_lst)
    plt.xlabel('EPISODES')
    plt.ylabel('AVG MAX Q VALUE')
    plt.ylim((-10, 1000))
    plt.savefig('graph_q')
    plt.close()

def plot_epsilon(episode_lst, epsilon_lst):
    plt.plot(episode_lst, epsilon_lst)
    plt.xlabel('EPISODES')
    plt.ylabel('EPSILON')
    plt.savefig('graph_epsilon')
    plt.close()

def get_td_error(agent_train, agent_target, transition, gamma):
    #transition = [state, action, reward, obs, done_mask]
    state, action, reward, obs, done_mask = transition
    state = torch.tensor(state, dtype=torch.float32)
    obs = torch.tensor(obs, dtype=torch.float32)

    target_action = agent_train(obs).argmax().item() # get max index
    td_target = reward + (gamma * agent_target(obs)[target_action] * done_mask)
    q_value = agent_train(state)[action]
    td_error = abs(td_target - q_value)

    return td_error