from DDQNAgent import DDQNAgent
import matplotlib.pyplot as plt

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
    plt.savefig('graph.png')