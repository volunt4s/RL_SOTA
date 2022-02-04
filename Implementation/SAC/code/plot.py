import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('Implementation/SAC/result/lunarlander/result.csv', delimiter=',', dtype=np.float32)
data2 = np.loadtxt('Implementation/TD3/result/result_lunar.csv', delimiter=',', dtype=np.float32)
episode_lst = data[0]
reward_lst = data[1]
episode_lst2 = data2[0]
reward_lst2 = data2[1]

df = pd.DataFrame(reward_lst)
a = df.rolling(24).mean()
a = a.values.tolist()
df2 = pd.DataFrame(reward_lst2)
a2 = df2.rolling(24).mean()
a2 = a2.values.tolist()

plt.plot(episode_lst, reward_lst, 'mistyrose')
plt.plot(episode_lst, reward_lst2, 'lightcyan')
plt.plot(episode_lst, a, 'red', label='SAC')
plt.plot(episode_lst, a2, 'deepskyblue', label='TD3')
plt.legend()

plt.ylabel('SCORE')
plt.xlabel('EPISODE')
plt.title('Lunarlander')

plt.savefig('score.png')
plt.close()