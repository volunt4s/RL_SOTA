import torch
import numpy as np
import collections
import gym
import random

def main():
    env = gym.make('CartPole-v1')
    env.reset()
    done = False
    a = [0, 1]

    while not done:
        env.render()
        obs, reward, done, _ = env.step(a[random.randint(0, 1)])

    print("finished")
    
if __name__ == "__main__":
    main()