import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('Implementation/DDPG/code/result.csv', delimiter=',', dtype=np.float32)
episode_lst = data[0]
reward_lst = data[1]


df = pd.DataFrame(reward_lst)
a = df.rolling(300).mean()
a = a.values.tolist()
plt.ylim((-10, 0))
plt.ylabel('SCORE')
plt.xlabel('EPISODE')
plt.title('MountainCarContinuous-v0')
plt.plot(episode_lst, reward_lst, 'lavender')
plt.plot(episode_lst, a, 'r')
plt.savefig('score.png')
plt.close()