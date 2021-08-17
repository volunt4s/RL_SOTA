import gym
import numpy
from DDQNAgent import DDQNAgent
from ReplayBuffer import ReplayBuffer

# HYPERPARAMETER
BATCH_SIZE = 3
EPISODES = 1

def main():
    env = gym.make('LunarLander-v2')
    agent = DDQNAgent()
    buffer = ReplayBuffer()

    for epi in range(EPISODES):
        state = env.reset()
        done = False
        reward_total = 0

        while not done:
            action = agent.sample_action(agent.numpy_to_torch(state), 0)
            obs, reward, done, _ = env.step(action)
            buffer.put([state, action, reward, obs, done])
            reward_total += reward
            state = obs


if __name__ == "__main__":
    main()
