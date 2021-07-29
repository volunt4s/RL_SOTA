import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# 학습에 필요한 하이퍼 파라미터
learning_rate = 0.0002
gamma = 0.99

class Policy(nn.Module):
    # Initializing, Fully connected layer 선언 4 -> 128 -> 2 (policy)
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    # Layer 거치는 과정 input -> relu -> softmax (classfication 위해)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax((self.fc2(x)), dim=0)
        return x

    def put_data(self, item):
        self.data.append(item)

    def train_net(self):
        # Return R
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    scores, episodes = [], []

    for n_epi in range(10000):
        s = env.reset()
        done = False


        while not done:
            if n_epi > 4000:
                env.render()

            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample() # Policy 확률분포에서 확률에 기반한 sample 뽑아냄
            s_prime, r, done, info = env.step(a.item())
            pi.put_data((r, prob[a]))
            s = s_prime
            score += r

        pi.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("EPISODE : {}, SCORE = {}".format(n_epi, score/print_interval))
            scores.append(score/print_interval)
            episodes.append(n_epi)
            plt.plot(episodes, scores)
            plt.xlabel('EPISODES')
            plt.ylabel('SCORES')
            plt.savefig('graph_reinforce')
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()

