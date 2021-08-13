import torch
import numpy as np
import collections
import gym
import random

def main():
    env = gym.make('LunarLander-v2')

    env.reset()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print("State : ", obs)
    print("Action : ", action)


if __name__ == "__main__":
    main()
