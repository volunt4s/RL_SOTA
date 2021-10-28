import torch
import numpy as np
import collections
import gym
import random

def main():
    env = gym.make('BipedalWalker-v3')

    state = env.reset()
    action = env.action_space.sample()
    done = False
    cnt = 0
    while not done:
        obs, reward, done, _ = env.step(action)
        state = obs
        cnt += 1
    print(cnt)

    
if __name__ == "__main__":
    main()
