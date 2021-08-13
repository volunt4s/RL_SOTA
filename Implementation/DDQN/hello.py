import gym

def main():
    env = gym.make('LunarLander-v2')

    env.reset()
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    print("State : ", obs)
    print("Action : ", action)


if __name__ == "__main__":
    main()
