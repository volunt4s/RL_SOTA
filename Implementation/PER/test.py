import numpy as np
import matplotlib.pyplot as plt

def main():
    num_lst = np.arange(101, dtype=np.float32)
    num_cnt = np.zeros((1, 101)).flatten()
    epsilon = 0.1
    # alpha 0 : uniform sampling, 1 : prioritized sampling
    alpha = 0.9

    priority = abs(num_lst) + epsilon
    priority = priority ** alpha

    probability = priority / np.sum(priority)
    # 1000 sampling
    for _ in range(1000):
        num_sample = sample(num_lst, probability)
        for idx in range(3):
            sampled = num_sample[idx]
            sampled = int(sampled) - 1
            num_cnt[sampled] += 1
    
    plt.bar(num_lst, num_cnt)
    plt.xlabel('NUMBER')
    plt.ylabel('NUMBER CNT')
    plt.ylim((0,100))
    plt.show()

def sample(list, probability):
    num_sample = np.random.choice(list, size=3, p=probability)
    return num_sample

if __name__ == "__main__":
    main()
