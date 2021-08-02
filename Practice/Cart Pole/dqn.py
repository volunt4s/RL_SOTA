import gym
import collections
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.0002
gamma = 0.99
buffer_limit = 50000
batch_size = 32


class ReplayBuffer():
    def __init__(self):
        # 저장할 버퍼 선언
        self.buffer = collections.deque(maxlen=buffer_limit)

    # 버퍼에 전이 상황 입력
    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        # 버퍼에서 n개 만큼 랜덤하게 샘플 추출
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            # 미니 배치안의 각 상황에 대해 모든 값 추출
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # 선언한 각 리스트를 입력하기 위한 텐서로 변환
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), torch.tensor(r_lst), torch.tensor(
            s_prime_lst, dtype=torch.float), torch.tensor(done_mask_lst)

    def size(self):
        return len(self.buffer)


class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Q 값은 음수가 나올 수 있으므로 Relu 거치면 안됨
        return x

    def sample_action(self, obs, epsilon):
        # Exploration criterion : epsilon-greedy
        # out -> neural net 거친 q value
        out = self.forward(obs)
        if random.random() < epsilon:
            return random.randint(0, 1)  # Epsilon만큼 랜덤하게 0, 1 행동
        else:
            return out.argmax().item()

    def outQvalue(self, obs):
        out = self.forward(obs)
        return out

def train(q, q_target, memory, optimizer):
    # 한 배치로 10번 학습
    for i in range(10):
        s, a, r, s_prime, done_mask = memory.sample(batch_size)

        # 샘플 배치 상태 s에서의 q value
        q_out = q(s)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # done_mask = 0 -> episode end => Reward만 받음
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    q = QNet()
    q_target = QNet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    scores, episodes = [], []
    print_interval = 20
    score = 0.0
    episodeCnt = 0
    qvalueTemp = np.array([0.0, 0.0])
    qvalueAvgPerEpi = []
    qvalueLeft_AvgPernEpi = []
    qvalueRight_AvgPernEpi = []

    qvalueDiff = []

    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    # 10000 episode 진행
    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s = env.reset()
        done = False

        # Terminate까지 계속 수행
        while not done:
            qvalueTemp += q.outQvalue(torch.from_numpy(s).float()).detach().numpy()
            # action from sample
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            # Action 수행
            s_prime, r, done, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            score += r
            episodeCnt += 1

            if done:
                qvalueTemp = qvalueTemp / episodeCnt
                qvalueAvgPerEpi.append([qvalueTemp[0], qvalueTemp[1]])
                episodeCnt = 0
                qvalueTemp = np.array([0.0, 0.0])
                break

        # 충분히 쌓일 때 까지 not train, 2000만큼 쌓이면 train 시작
        if memory.size() > 2000:
            train(q, q_target, memory, optimizer)

        if n_epi % print_interval == 0 and n_epi != 0:
            # 20번 에피소드마다 q_target update
            q_target.load_state_dict(q.state_dict())

            print("EPISODE : {}, SCORE : {:.1f}, BUFFER : {}, EPSILON : {:.1f}%".format(n_epi, score / print_interval,
                                                                                           memory.size(), epsilon * 100))
            qvalueAvg = np.sum(qvalueAvgPerEpi, axis=0) / print_interval
            qvalueLeft_AvgPernEpi.append(qvalueAvg[0] * 1000)
            qvalueRight_AvgPernEpi.append(qvalueAvg[1] * 1000)
            episodes.append(n_epi)

            plt.plot(episodes, qvalueLeft_AvgPernEpi, 'b')
            plt.plot(episodes, qvalueRight_AvgPernEpi, 'g')
            plt.xlabel('EPISODES')
            plt.ylabel('AVG Q VALUE')
            plt.savefig('graph_qvalue2.png')
            qvalueAvgPerEpi.clear()

#            scores.append(score/print_interval)
#            episodes.append(n_epi)
#            plt.plot(episodes, scores)
#            plt.xlabel('EPISODES')
#            plt.ylabel('SCORES')
#            plt.savefig('graph_dqn.png')
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
