import gym
env = gym.make("BipedalWalkerHardcore-v3")

state = env.reset()
print(env.action_space.sample())
