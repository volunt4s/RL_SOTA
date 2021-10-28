import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.loadtxt('output_data.csv', delimiter=',', dtype=np.float32)
episode_lst = data[0]
reward_lst = data[1]
qvalue_lst = data[2]
epsilon_lst = data[3]

df = pd.DataFrame(reward_lst)
a = df.rolling(30).mean()
a = a.values.tolist()
plt.ylim((-500, 300))
plt.ylabel('SCORE')
plt.xlabel('EPISODE')
plt.plot(episode_lst, reward_lst, 'lavender')
plt.plot(episode_lst, a, 'r')
plt.savefig('score.png')
plt.close()

df = pd.DataFrame(qvalue_lst)
a = df.rolling(30).mean()
a = a.values.tolist()
plt.ylim((-10, 5000))
plt.ylabel('MAX Q VALUE')
plt.xlabel('EPISODE')
plt.plot(episode_lst, qvalue_lst, 'lavender')
plt.plot(episode_lst, a, 'r')
plt.savefig('qvalue.png')
plt.close()