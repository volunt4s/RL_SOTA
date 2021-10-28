import gym
import torch
import torch.optim as optim
import csv
from PPO import PPO
from Buffer import Buffer
from torch.distributions import Normal
from train import train

EPISODES = 30000
LEARNING_RATE = 0.0001

def main():
    env = gym.make('LunarLanderContinuous-v2')
    model = PPO()
    buffer = Buffer()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    episode_lst, score_lst = [], []

    for episode in range(EPISODES):
        score = 0
        state = env.reset()
        done = False
        while not done:
            for i in range(10):
                mu, std = model.pi(torch.from_numpy(state).float())
                dist = Normal(mu, std)
                action = dist.sample()
                old_log_prob = dist.log_prob(action)
                obs, reward, done, _ = env.step(action.numpy())
                
                done_mask = 0.0 if done else 1.0
                transition = [state, action.numpy(), reward, obs, done_mask, old_log_prob.detach().numpy()]
                buffer.put(transition)
                
                state = obs
                score += reward

                if done: break
            train(buffer, model, optimizer)
        
        print("EPISODE : {:d}\tSCORE : {:.1f}".format(episode, score))
        episode_lst.append(episode)
        score_lst.append(score)

    # Save trained model
    torch.save(model.state_dict(), "trained_model")
    # Save result csv (episode, score)
    f = open('result.csv', 'w', newline='')
    w = csv.writer(f)
    w.writerow(episode_lst)
    w.writerow(score_lst)
    f.close()


if __name__ == "__main__":
    main()