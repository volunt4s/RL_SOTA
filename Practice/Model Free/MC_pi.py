'''
Evaluating PI
Using first quadrant -> [0,1]
circle area 1 = pi * r^2 / 4
    => pi = 4 * ratio of circle area
dart ramdomly in range [0,1] to use monte carlo
'''

import math as m
import random as r

# Total number of iterate
ITERATION = 10000000
# Total number of inside circle
INSIDE = 0

for i in range(0, ITERATION):
    x = r.random() ** 2
    y = r.random() ** 2
    radius = m.sqrt(x + y)
    if radius < 1:
        INSIDE += 1

pi = 4 * (float(INSIDE) / ITERATION)

print("INSIDE : ", INSIDE)
print("PI : ", pi)
