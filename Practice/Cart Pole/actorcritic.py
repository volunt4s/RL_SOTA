import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.0002
gamma = 0.99
n_rollout = 10

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_v = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc_pi.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)

    # Neural net 거쳐 Pi return
    def pi(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    # Neural net 거쳐 value function return
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for transition in self.data:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)
        done_batch = torch.tensor(done_lst, dtype=torch.float)

        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train_net(self):
        s, a, r, s_prime, done = self.make_batch()
        td_target = r + gamma * self.v(s_prime) * done
        delta = td_target - self.v(s)

        pi = self.pi(s, softmax_dim=1)
        pi_a = pi.gather(1, a)
        loss = -torch.log(pi_a) * delta.detach() + F.smooth_l1_loss(self.v(s), td_target.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def return_vauleNet(self, obs):
        valueNet = self.forward(obs)
        return valueNet.detach().numpy()


def main():
    env = gym.make('CartPole-v1')
    model = ActorCritic()
    print_interval = 20
    score = 0.0
    value_list_max = []

    # Matplotlib params
    plt_scores, plt_episodes, plt_values = [], [], []

    for n_epi in range(10000):
        done = False
        # Value params
        value_list = []

        s = env.reset()
        while not done:
            for t in range(n_rollout):
                prob = model.pi(torch.from_numpy(s).float())
                m = Categorical(prob)
                a = m.sample().item()
                v = model.v(torch.from_numpy(s).float()).detach().numpy()
                value_list.append(abs(v[0]))

                s_prime, r, done, info = env.step(a)
                model.put_data((s, a, r, s_prime, done))
                s = s_prime
                score += r

                if done:
                    break

            if abs(v[0]) < 1.25:
                model.train_net()

        value_max = max(value_list)
        #print("Max value in {:f} epi : {:f}".format(n_epi, value_max))
        value_list_max.append(value_max) # 0 ~ n_epi

        if n_epi % print_interval == 0 and n_epi != 0:
            print("EPISODE : {}, SCORE : {:.1f}".format(n_epi, score/print_interval))
            plt_scores.append(score / print_interval)
            plt_episodes.append(n_epi)
            score = 0.0

    # Plotting Data
    plt.plot(plt_episodes, plt_scores)
    plt.xlabel('EPISODES')
    plt.ylabel('SCORES')
    plt.savefig('graph_score.png')
    plt.close()

    plt.plot(range(0, 10000), value_list_max)
    plt.xlabel('EPISODES')
    plt.ylabel('Max abs(value) each episode')
    plt.savefig('graph_value.png')
    env.close()


if __name__ == '__main__':
    main()