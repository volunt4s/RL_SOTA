import gym
import torch
from DDPG import Actor

TEST_EPISODES = 100

def main():
    env = gym.make("Pendulum-v0")
    actor = Actor()

    actor.load_state_dict(torch.load('Implementation/DDPG/code/trained_actor'))
    
    for episode in range(TEST_EPISODES):
        state = env.reset()
        env.render()
        done = False
        while not done:
            action = actor(torch.from_numpy(state).float())
            action = action.item()
            next_state, reward, done, _ = env.step([action])
            state = next_state
    

    env.close()
if __name__ == "__main__":
    main()