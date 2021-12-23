import gym
import torch
import torch.optim as optim
import numpy as np
import csv
from DDPG import Actor, Critic
from ReplayMemory import ReplayMemory
from noise import OrnsteinUhlenbeckProcess
from train import train
from utils import soft_update

#HYPERPARMS
EPISODE = 50000
BUFFER_SIZE = 100000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.01

def main():
    env = gym.make("MountainCarContinuous-v0")
    actor = Actor()
    critic = Critic()
    actor_target = Actor()
    critic_target = Critic()
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=0.0001)
    critic_optim = optim.Adam(critic.parameters(), lr=0.001)
    memory = ReplayMemory(BUFFER_SIZE, BATCH_SIZE)

    episode_lst = []
    score_lst = []

    for episode in range(EPISODE):
        state = env.reset()
        done = False
        step = 1
        score = 0.0
        noise = OrnsteinUhlenbeckProcess(mu=np.zeros(1))

        while not done:
            action = actor(torch.from_numpy(state).float())
            print(state.shape)
            action = action.item() + noise()[0]
            next_state, reward, done, _ = env.step([action])
            done_mask = 0.0 if done else 1.0
            transition = [state, action, reward/100.0, next_state, done_mask]
            memory.put(transition)

            state = next_state
            step += 1
            score += reward
            
            if memory.size() > 10000:
                if step % 10 == 0 and step != 0:
                    train(actor, actor_target, critic, critic_target, memory, GAMMA, actor_optim, critic_optim)
                if step % 100 == 0 and step != 0:
                    soft_update(actor_target, actor, TAU)
                    soft_update(critic_target, critic, TAU)
        
        episode_lst.append(episode)
        score_lst.append(score)

        if episode % 20 == 0:
            print("EPISODE : {:d}, REWARD : {:.2f}, STEP : {:d}".format(episode, score, step))
            
    torch.save(actor.state_dict(), "trained_actor")
    torch.save(critic.state_dict(), "trained_critic")

    # Save result csv (episode, score)
    f = open('result.csv', 'w', newline='')
    w = csv.writer(f)
    w.writerow(episode_lst)
    w.writerow(score_lst)
    f.close()
    env.close()

if __name__ == "__main__":
    main()
