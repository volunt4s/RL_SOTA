import gym
import torch.optim as optim
import torch
import csv
from DDQNAgent import DDQNAgent
from ReplayBuffer import ReplayBuffer
from train import *
from utils import *

# HYPERPARAMETER
BATCH_SIZE = 32
BUFFER_MAXLEN = 10000
EPISODES = 600
GAMMA = 0.99
LEARNING_RATE = 0.001
TRAIN_RATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.08
EPSILON_DECAY = 0.995
EPSILON_SAMPLE = 0.01
WEIGHT_DECAY = 0.001
ALPHA = 0.3
BETA_START = 0.5
BETA_END = 1
BETA_RATE = 1.001

def main():
    env = gym.make('LunarLander-v2')
    agent_train = DDQNAgent()
    agent_target = DDQNAgent()
    agent_target.load_state_dict(agent_train.state_dict())
    buffer = ReplayBuffer(BUFFER_MAXLEN, EPSILON_SAMPLE, ALPHA, BETA_START)
    optimizer = optim.Adam(agent_train.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    epsilon = EPSILON_START
    episode_lst, reward_lst, qvalue_lst, epsilon_lst = [], [], [], []

    for epi in range(EPISODES):
        state = env.reset()
        done = False
        reward_total = 0
        qvalue_total = 0
        step_cnt = 0

        while not done:
            action = agent_train.sample_action(agent_train.numpy_to_torch(state), epsilon)
            # Compute Q value
            qvalue_out = agent_train.out_qvalue(agent_train.numpy_to_torch(state)).detach().numpy()
            qvalue_max = qvalue_out.max()
            qvalue_total += qvalue_max

            obs, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0

            transition = [state, action, reward, obs, done_mask]
            td_error = get_td_error(agent_train, agent_target, transition, gamma=GAMMA)
            buffer.put([td_error, transition])

            reward_total += reward
            state = obs
            step_cnt += 1
            # train for each step and update
            if buffer.size() > 5000:
                if step_cnt % 10 and step_cnt != 0:
                    train_ddqn(agent_train, agent_target, buffer, optimizer, BATCH_SIZE, GAMMA)

                if step_cnt % 20 and step_cnt != 0:
                    load_network(agent_train, agent_target)

        print("EPISODES : {:d}, REWARD : {:.1f}, EPSILON : {:.1f}%, BUFFER_SIZE : {:d} STEP CNT : {:d}, BETA : {:.3f}".format(epi, reward_total, epsilon*100, buffer.size(), step_cnt, buffer.get_beta()))
        
        epsilon = decaying_eps(epsilon, EPSILON_END, EPSILON_DECAY)
        buffer.improving_beta(BETA_END, BETA_RATE)

        # append list to plot
        episode_lst.append(epi)
        reward_lst.append(reward_total)
        qvalue_lst.append(qvalue_total / float(step_cnt))
        epsilon_lst.append(epsilon*100)
    
    with open('output_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(episode_lst)
        writer.writerow(reward_lst)
        writer.writerow(qvalue_lst)
        writer.writerow(epsilon_lst)

    plot_reward(episode_lst, reward_lst)
    plot_qvalue(episode_lst, qvalue_lst)
    plot_epsilon(episode_lst, epsilon_lst)

    # Save trained model
    torch.save(agent_train.state_dict(), "trained_model")
    env.close()

if __name__ == "__main__":
    main()
