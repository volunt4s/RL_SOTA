import gym
import numpy as np
from TD3 import TD3
from ReplayMemory import ReplayMemory
from utils import save_csv

#HYPERPARMS
EPISODE = 1000
BUFFER_SIZE = 30000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.01
POLICY_NOISE = 0.2
NOISE_CLIP = 0.4

episode_lst = []
score_lst = []

def evaluation(policy, current_episode):
    eval_env = gym.make("BipedalWalkerHardcore-v3")
    score = 0.0
    state = eval_env.reset()
    done = False

    while not done:
        action = policy.select_action(np.array(state))
        action = action.detach().numpy()            
        state, reward, done, _ = eval_env.step(action)
        score += reward
    
    score_lst.append(score)
    episode_lst.append(current_episode)

    print("EPISODE : {:d}, Current policy SCORE : {:.6f}".format(current_episode, score))

def main():
    env = gym.make("BipedalWalker-v3")
    policy = TD3(gamma=GAMMA, tau=TAU, policy_noise=POLICY_NOISE, noise_clip=NOISE_CLIP)
    replay_buffer = ReplayMemory(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    

    for episode in range(EPISODE):
        state = env.reset()
        done = False
        step = 1
        

        while not done:
            step += 1
            if step < 10:
                action = env.action_space.sample()
            else :
                action = policy.select_action(state.flatten())
                action = action.detach().numpy() + np.random.normal(0, 0.2, size=env.action_space.shape[0])
            next_state, reward, done, _ = env.step(action)
            done_mask = 0.0 if done else 1.0
            transition = [state, action, reward, next_state, done_mask]
            replay_buffer.put(transition)

            state = next_state
            
            if replay_buffer.size() > 10000:
                policy.train(replay_buffer)

        # Evaluate policy
        evaluation(policy, episode)

    policy.save()
    save_csv(episode_lst, score_lst)
        

if __name__ == "__main__":
    main()
