

from __future__ import print_function, division
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, norm

np.random.seed(1)
bandit_means = [1,2,3]
num_trail = 2000


class Bandit:
    def __init__(self, m_0):
        self.m_0 = m_0
        self.tau = 1
        #initialize prim_0or to norm(0,1), which later will become posterior
        self.m = 0
        self.lambda_ = 1
        self.sum_x = 0 #just for easy to compute
        self.N = 0
    
    def pull(self):
        return np.random.randn()/np.sqrt(self.tau) + self.m_0
    
    def sample(self):
        # the prior mean is 0 and precison is 1, so formula will be below
        return np.random.randn()/np.sqrt(self.lambda_ ) + self.m
    
    def update(self, x):
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = self.tau*self.sum_x/self.lambda_
        self.N += 1

def plot(bandits,trail):
    x = np.linspace(-3,6,200)
    for b in bandits:
        y = norm.pdf(x, b.m, np.sqrt(1. /b.lambda_))
        #norm, mean/standard deviation
        plt.plot(x, y, label = f"Real Mean: {b.m_0:.4f}, num plays = {b.N}")
    plt.title(f"Bandit distributions after {trail} trails")
    plt.legend()
    plt.show()

def experiment():
    bandits = [Bandit(m) for m in bandit_means]
    sample_points = [5,10,20,50,100,200,500,1999]
    rewards = np.empty(num_trail)
    
    for i in range(num_trail):
        #Thompson sampling
        j = np.argmax([m.sample() for m in bandits])

        #posterior
        if i in sample_points:
            plot(bandits,i)

        #pull
        x = bandits[j].pull()

        #update rewards
        rewards[i] = x

        #update the distribution for the bandit pulled
        bandits[j].update(x)

    cum_avg = np.cumsum(rewards)/np.arange(num_trail)+1

    #plot moving avg ctr
    plt.plot(cum_avg)
    for m in bandit_means:
        plt.plot(np.ones(num_trail)*m)
    plt.show()

if __name__ == "__main__":
    experiment()