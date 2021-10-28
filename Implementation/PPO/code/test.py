import gym
import torch
from PPO import PPO
from torch.distributions import Categorical

TEST_EPISODES = 100

def main():
    env = gym.make('LunarLander-v2')
    model_trained = PPO()
    model_trained.load_state_dict(torch.load('Implementation/PPO/trained_model'))
    
    for epi in range(TEST_EPISODES):
        done = False
        state = env.reset()
        reward_sum = 0
        while not done:
            env.render()
            probs = model_trained.pi(torch.from_numpy(state).float())
            m = Categorical(probs)
            action = m.sample().item()
            obs, reward, done, _ = env.step(action)
            state = obs
            reward_sum += reward

        print("REWARD : ", reward_sum)
    env.close()


if __name__ == "__main__":
    main()