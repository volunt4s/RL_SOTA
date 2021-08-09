import gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("CarRacing-v0")

env.reset()
screen = env.render('rgb_array')
print(screen.shape)
plt.figure()
plt.imshow(screen)
plt.show()
env.close()
