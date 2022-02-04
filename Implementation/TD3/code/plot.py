import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('Implementation/TD3/result/result_bipedal_hard.csv', delimiter=',', dtype=np.float32)
episode_lst = data[0]
reward_lst = data[1]


df = pd.DataFrame(reward_lst)
a = df.rolling(24).mean()
a = a.values.tolist()
plt.ylabel('SCORE')
plt.xlabel('EPISODE')
plt.title('BipedalWalker (Hard)')
plt.plot(episode_lst, reward_lst, 'lavender')
plt.plot(episode_lst, a, 'r')
plt.savefig('score.png')
plt.close()