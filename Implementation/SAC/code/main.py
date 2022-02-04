import gym
import numpy as np
from ReplayBuffer import ReplayBuffer
from utils import save_csv
from SAC import SAC

#HYPERPARMS
EPISODE = 1000
BUFFER_SIZE = 50000
BATCH_SIZE = 128
GAMMA = 0.98
TAU = 0.02
LR_ACTOR = 5e-4
LR_CRITIC = 1e-3
LR_ALPHA = 1e-3
INIT_ALPHA = 0.01

episode_lst = []
score_lst = []

def evaluation(policy, current_episode):
    eval_env = gym.make("LunarLanderContinuous-v2")
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

    print("EPISODE : {:d}, Current policy SCORE : {:.6f}, Alpha : {:.4f}".format(current_episode, score, policy.get_alpha()))

def main():
    env = gym.make("LunarLanderContinuous-v2")
    policy = SAC(gamma=GAMMA, tau=TAU, lr_actor=LR_ACTOR, lr_critic=LR_CRITIC, init_alpha=INIT_ALPHA, lr_alpha=LR_ALPHA)
    replay_buffer = ReplayBuffer(buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE)
    
    for episode in range(EPISODE):
        state = env.reset()
        done = False
        
        while not done:
            action = policy.select_action(state.flatten())
            action = action.detach().numpy()            
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
