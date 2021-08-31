import numpy as np
import collections

a = np.array([1, 3, 2, 5, 4])
rank = a.argsort() + 1
print(rank)