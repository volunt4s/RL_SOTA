import gym
import torch
import torch.optim as optim
import csv
from PPO import PPO
from Buffer import Buffer
from train import train
from torch.distributions import Categorical
# Hyperparameter
EPISODES = 10000
LEARNING_RATE = 0.0001

def main():
    model = PPO()
    buffer = Buffer()
    env = gym.make('LunarLander-v2')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    episode_lst, score_lst = [], []

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        score = 0.0

        while not done:
            for i in range(50):
                probs = model.pi(torch.from_numpy(state).float())
                m = Categorical(probs)
                action = m.sample().item()
                obs, reward, done, _ = env.step(action)
                done_mask = 0.0 if done else 1.0

                transition = [state, action, reward, obs, done_mask, probs[action].item()]
                buffer.put(transition)
                score += reward
                state = obs
                if done:
                    break

            train(buffer, model, optimizer)
        
        episode_lst.append(episode)
        score_lst.append(score)
        print("EPISODE : {:d}\tSCORE : {:.2f}".format(episode, score))

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