import numpy as np

class OrnsteinUhlenbeckProcess():
    """ 
    Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, mu):
        self.theta = 0.15
        self.dt = 0.01
        self.sigma = 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x