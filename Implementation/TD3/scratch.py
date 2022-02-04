import gym
env = gym.make("LunarLanderContinuous-v2")

state = env.reset()
print(env.action_space.sample())
print(state.shape)
