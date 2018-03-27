"""
Perform EM on a given input file and finds 3 clusters from the data
"""

#%%
import csv
import scipy.stats as stats
import numpy as np
from typing import Sequence, Tuple

class dist(object):
    def __init__(self, mean_x, var_x, mean_y, var_y, fraction_of_total):
        self.mean_x = mean_x
        self.mean_y = mean_y

        self.var_x = var_x
        self.var_y = var_y

        self.fraction_of_total = fraction_of_total
    
    def responsibility(self, coord: Tuple[int, int]) -> float: 
        """
        calculate responsibility for a single point

        args: 
            coord: (x,y)

        returns:
            float probability it is from this distribution
        """
        cov = [[self.var_x, 0], [0, self.var_y]]
        # Note: prob is really small, but shouldn't matter after you normalize
        return stats.multivariate_normal([self.mean_x, self.mean_y], cov=cov).pdf(coord)


    
    pass

def load_data(): 
    """
    return a list of [(x,y)]
    """
    coords = []
    with open("./sample.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            coords.append((float(row[0]), float(row[1])))
        
    return coords

def calc_responsibility(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    """
    args: 
        points: matrix for [X,Y] each row is point
    
    returns:    
        matrix of R x C R is each point, C is each distribution
    """
    # Going to use a stupid for loop
    total_probs = []
    for point in points: 
        probs = []
        print(point)
        for c in dists:
            print(c)
            r = c.responsibility(point)
            print(f"prob: {r}")
            probs.append(r)
        total_probs.append(probs)

    return total_probs

#%% TESTING
data = load_data()

import matplotlib.pyplot as plt
def plot(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.scatter(x,y)

dis = dist(1, 1, 1, 1, 1)
print(dis.responsibility((1,1)))
resp = calc_responsibility([[1,2],[1,1], [2,2]], [dist(1,1, 1, 1, 1), dist(2,1, 2, 1, 1)])
for row in resp: 
    print(row)
