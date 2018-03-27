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
    
    def __str__(self):
        return f"xm: {self.mean_x}, xv: {self.var_x}, ym: {self.mean_y}, yv: {self.var_y}, frac: {self.fraction_of_total}"

    def __repr__(self):
        return str(self)
    
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
        return self.fraction_of_total * stats.multivariate_normal([self.mean_x, self.mean_y], cov=cov).pdf(coord)


    
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
        for c in dists:
            r = c.responsibility(point)
            probs.append(r)
        total_probs.append(probs)

    return np.array(total_probs)

def update_dists(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    """
    Update the location of the given distributions based on their 'responsibility' 
    for each point
    """
    resp = calc_responsibility(points, dists)
    total_weight = np.sum(resp)
    updated_dists = []
    for c, d in enumerate(dists): 
        weight = np.sum(resp[:, c])
        rel_weight = weight / total_weight
        new_mean_x = 0.0
        new_mean_y = 0.0
        for r, point in enumerate(points): 
            new_mean_x += resp[r][c] * point[0]
            new_mean_y += resp[r][c] * point[1]
        
        new_var_x = 0.0
        new_var_y = 0.0
        for r, point in enumerate(points): 
            new_var_x += resp[r][c]  * (point[0] - new_mean_x)**2
            new_var_y += resp[r][c]  * (point[1] - new_mean_y)**2
            

        new_dist = dist(new_mean_x / weight, new_var_x / weight, new_mean_y / weight, new_var_y / weight, rel_weight)
        updated_dists.append(new_dist)
    return updated_dists


#%% TESTING
data = load_data()

import matplotlib.pyplot as plt
def plot(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.scatter(x,y)
    plt.show()


dists = [dist(0,10,0, 10, .4), dist(3,10,0, 10, .3), dist(10,30,10, 10, .3)]

for i in range(100): 
    dists = update_dists(data, dists)

for d in dists:
    print(d)
plot(data)