##Thompson Sampling is nice becasue the suboptimal bandit is only pushed down enough so that 
## their posterior is hanving very little probability mass beyond the peak of the optimal band.
## The Key is that they remain fat posterior distributions -- explore just enough to be very confident
## that their means are no better than the optimal band. (Computational efficiency?)

from __future__ import print_function, division
from builtins import range

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

np.random.seed(2)
bandit_prob = [0.2,0.5,0.75]
num_trail = 2000



class Bandit:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1
        self.N = 0
    
    def pull(self):
        return np.random.random() < self.p
    
    def sample(self):
        return np.random.beta(self.a, self.b)
    
    def update(self, x):
        self.a += x
        self.b += 1-x
        self.N += 1

def plot(bandits,trail):
    x = np.linspace(0,1,200)
    for b in bandits:
        y = beta.pdf(x, b.a, b.b)
        plt.plot(x, y, label = f"real p: {b.p:.4f}, win rate = {b.a -1}/{b.N}")
    plt.title(f"Bandit distributions after {trail} trails")
    plt.legend()
    plt.show()

def experiment():
    bandits = [Bandit(p) for p in bandit_prob]
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

    print("total reward earned:", rewards.sum())
    print("overall win rate:", rewards.sum()/num_trail)
    print("num times selected each bandit:", [b.N for b in bandits])

if __name__ == "__main__":
    experiment()