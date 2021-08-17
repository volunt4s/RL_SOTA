import gym
from torch._C import AggregationType
import torch.optim as optim
import torch
from DDQNAgent import DDQNAgent
from ReplayBuffer import ReplayBuffer
from train import train
from common_functions import decaying_eps
from common_functions import load_network
from common_functions import plot_reward

# HYPERPARAMETER
BATCH_SIZE = 32
BUFFER_MAXLEN = 10000
EPISODES = 500
GAMMA = 0.99
LEARNING_RATE = 0.001
TRAIN_RATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999

def main():
    env = gym.make('LunarLander-v2')
    agent_train = DDQNAgent()
    agent_target = DDQNAgent()
    agent_target.load_state_dict(agent_train.state_dict())
    buffer = ReplayBuffer(BUFFER_MAXLEN)
    optimizer = optim.Adam(agent_train.parameters(), lr=LEARNING_RATE)
    epsilon = EPSILON_START
    episode_lst, reward_lst = [], []

    for epi in range(EPISODES):
        state = env.reset()
        done = False
        reward_total = 0
        step_cnt = 0
        while not done:
            if epi > 500:
                env.render()
            action = agent_train.sample_action(agent_train.numpy_to_torch(state), epsilon)
            obs, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            buffer.put([state, action, reward, obs, done_mask])
            reward_total += reward
            state = obs
            step_cnt += 1
                
            # train TRAIN_RATE times for each episode
            if buffer.size() > 5000:
                if step_cnt % 10 and step_cnt != 0:
                    train(agent_train, agent_target, buffer, optimizer, BATCH_SIZE, GAMMA)
                if step_cnt % 20 and step_cnt != 0:
                    load_network(agent_train, agent_target)
                    epsilon = decaying_eps(epsilon, EPSILON_END, EPSILON_DECAY)
            
        print("EPISODES : {:d}, REWARD : {:.1f}, EPSILON : {:.1f}%, BUFFER_SIZE : {:d} STEP CNT : {:d}".format(epi, reward_total, epsilon*100, buffer.size(), step_cnt))
        episode_lst.append(epi)
        reward_lst.append(reward_total)
    
    plot_reward(episode_lst, reward_lst)
    torch.save(agent_train.state_dict(), "trained_model")
    env.close()

if __name__ == "__main__":
    main()
