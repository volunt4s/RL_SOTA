import torch
import numpy as np
import collections
import gym
import random

def main():
    env = gym.make('CarRacing-v0')

    state = env.reset()
    action = env.action_space.sample()
    print(state.shape)


if __name__ == "__main__":
    main()
