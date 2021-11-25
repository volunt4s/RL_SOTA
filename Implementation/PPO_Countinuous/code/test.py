import gym
import torch
from PPO import PPO
from torch.distributions import Normal

TEST_EPISODES = 100

def main():
    env = gym.make('BipedalWalker-v2')
    model = PPO()
    model.load_state_dict(torch.load('Implementation/PPO_Countinuous/trained_model_lunar'))
    
    for epi in range(TEST_EPISODES):
        done = False
        state = env.reset()
        reward_sum = 0
        while not done:
            env.render()
            mu, std = model.pi(torch.from_numpy(state))
            dist = Normal(mu, std)
            action = dist.sample()
            obs, reward, done, _ = env.step(action.numpy())
            state = obs
            reward_sum += reward

        print("REWARD : ", reward_sum)
    env.close()


if __name__ == "__main__":
    main()