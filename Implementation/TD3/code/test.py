import gym
import torch
from TD3 import TD3

TEST_EPISODES = 100
BUFFER_SIZE = 20000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.01
POLICY_NOISE = 0.1
NOISE_CLIP = 0.4

def main():
    env = gym.make("BipedalWalker-v3")
    policy = TD3(gamma=GAMMA, tau=TAU, policy_noise=POLICY_NOISE, noise_clip=NOISE_CLIP)
    policy.actor.load_state_dict(torch.load('Implementation/TD3/trained_actor'))
    
    for episode in range(TEST_EPISODES):
        state = env.reset()
        done = False
        score = 0.0

        while not done:
            env.render()
            action = policy.select_action(state.flatten())
            action = action.detach().numpy()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            score += reward
        
        print("SCORE : ", score)

if __name__ == "__main__":
    main()