from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

num_trails = 100000
eps = 0.1
bandit_prob = [0.2,0.5,0.75]

class Bandit:
    def __init__(self,p):
        self.p = p
        self.p_estimate = 0.
        self.N = 0

    def pull(self):
        #?
        return np.random.random() < self.p
    
    def update(self, x):
        self.N += 1.
        self.p_estimate = ((self.N-1)*self.p_estimate+x)/self.N
    

def ucb(mean, n, nj):
    return mean + np.sqrt(2*np.log(n)/nj)

def run_experiment():
    bandits = [Bandit(p) for p in bandit_prob]
    reward = np.empty(num_trails)
    total_plays = 0
    
    #initialize the play, deal with infinity
    for j in range(len(bandits)):
        x = bandits[j].pull()
        total_plays += 1
    
    for i in range(num_trails):
        j = np.argmax([ucb(b.p_estimate, total_plays,b.N) for b in bandits])
        x = bandits[j].pull()
        total_plays += 1
        bandits[j].update(x)
        reward[i] = x
    
    cum_avg = np.cumsum(reward)/(np.arange(num_trails)+1)

    #plot moving average ctr
    plt.plot(cum_avg)
    plt.plot(np.ones(num_trails)*np.max(bandit_prob))
    plt.xscale('log')
    plt.show()


    #plot moving average ctr linear
    plt.plot(cum_avg)
    plt.plot(np.ones(num_trails)*np.max(bandit_prob))
    plt.show()

    for b in bandits:
        print(b.p_estimate)

    print("total reward earned:", reward.sum())
    print("overall win rate:", reward.sum() / num_trails)
    print("num times selected each bandit:", [b.N for b in bandits])
    
    return cum_avg

if __name__ == '__main__':
  run_experiment()

