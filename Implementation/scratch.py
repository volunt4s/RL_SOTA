import math
import numpy as np

def findwn(x1, x2, d) :
    t = 0.1357
    tau_d = t * d
    wd = 2*math.pi / tau_d
    delta = np.log(x1/x2)
    zeta = delta / (np.sqrt((2*math.pi)**2 + delta**2))
    wn = wd / np.sqrt(1 - (zeta**2))

    print("wn = ", wn)

def main():
    findwn(x1=2.4, x2=1.5, d=3.3)

if __name__ == "__main__":
    main()