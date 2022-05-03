from genericpath import exists
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import os

class Plot():
    def __init__(self):
        self.parent_dir = 'output'
        os.makedirs(self.parent_dir, exist_ok=True)
        print("output made")

    def __call__(self, results, interval):
        self.save_results(self.parent_dir,results,interval)
    
    def save_results(fn, results, interval):

        y = np.mean(results, axis=0)
        error = np.std(results, axis=0)

        x = range(0, results.shape[1]*interval, interval)
        fig, ax = plt.subplots(1,1,figsize=(6,5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x,y,yerr=error,fmt='-o')
        plt.savefig(fn+'training_results.png')
        savemat(fn+'training_results.mat', {'reward':results})


