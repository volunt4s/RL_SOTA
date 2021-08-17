import gym
import torch
from DDQNAgent import DDQNAgent

TEST_EPISODES = 100

def main():
    env = gym.make('LunarLander-v2')
    agent_trained = DDQNAgent()
    agent_trained.load_state_dict(torch.load('Implementation/DDQN/trained model/trained_model'))
    
    for epi in range(TEST_EPISODES):
        done = False
        state = env.reset()
        while not done:
            env.render()
            action = agent_trained.sample_action(agent_trained.numpy_to_torch(state), 0)
            obs, reward, done, _ = env.step(action)
            state = obs

    env.close()


if __name__ == "__main__":
    main()