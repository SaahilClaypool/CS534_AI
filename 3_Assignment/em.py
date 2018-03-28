"""
Perform EM on a given input file and finds 3 clusters from the data
"""

#%%
import csv
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import pandas as pd
import random
import math
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
    coords = np.empty([0,2])
    with open("./sample.csv") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            coords = np.append(coords, [[float(row[0]), float(row[1])]], axis=0)
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
        total_probs.append(probs / sum(probs))

    return np.array(total_probs)

def calc_log_likelihood(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    """
    args:
        points: matrix for [X,Y] each row is point

    returns:
        total sum of log(r1+r2+...) where rn is the responsibility computed by each dist
    """
    log_like = 0
    for point in points:
        p_sum = 0
        for c in dists:
            r = c.responsibility(point)
            p_sum += r
        log_like += np.log(p_sum)
    return log_like

def update_dists(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    """
    Update the location of the given distributions based on their 'responsibility' 
    for each point
    """
    resp = calc_responsibility(points, dists)
    total_weight = np.sum(resp)
    updated_dists = []
    error_var = 0
    for c, d in enumerate(dists): 
        mc = np.sum(resp[:, c]) # total amount of responsibility for this cluster
        fraction_of_total = mc / len(points) # normalize by number of points
        new_mean_x = 0.0
        new_mean_y = 0.0
        for r, point in enumerate(points):
            new_mean_x += resp[r][c] * point[0]
            new_mean_y += resp[r][c] * point[1]
        
        new_mean_x = new_mean_x / mc
        new_mean_y = new_mean_y / mc
        
        new_var_x = 0.0
        new_var_y = 0.0
        for r, point in enumerate(points): 
            new_var_x += resp[r][c]  * (point[0] - new_mean_x)**2
            new_var_y += resp[r][c]  * (point[1] - new_mean_y)**2
        new_var_x = new_var_x / mc
        new_var_y = new_var_y / mc

        new_dist = dist(new_mean_x , new_var_x , new_mean_y , new_var_y , fraction_of_total)
        updated_dists.append(new_dist)
    # print(compute_BIC(points, dists))
    return updated_dists

def find_clusters(points: Sequence[Tuple[int, int]], number: int, restarts: int = 0, iterations = 75) -> \
        Tuple[Sequence[dist], float]:
    """
    Find the best model with the given number of clusters and restarts
    """
    if (restarts < 1): restarts = 1
    best_model: Sequence[dist]
    best_likelihood = -math.inf
    for r in range(restarts):
        print(f"restart: {r}")
        dists = init_clusters(number)
        for i in range(iterations):
            dists = update_dists(points, dists)

        # plot_clusters(points, calc_responsibility(points, dists))
        likeli = calc_log_likelihood(points, dists)
        if (likeli >= best_likelihood):
            best_likelihood = likeli
            best_model = dists

    return (best_model, best_likelihood)

def find_number_of_clusters(points: Sequence[Tuple[int, int]], restarts: int = 0, iterations = 75) -> Tuple[Sequence[dist], float]:
    """
    Find the best model by determining the best number of clusters
    """
    #default restarts is 1
    if (restarts < 1): restarts = 1
    smallest_BIC = math.inf
    best_model: Sequence[dist]
    best_model_likelihood = 0
    best_n = 0

    #our model should have, at most, the same number of clusters as points
    for i in range(len(points)):
        model, likelihood = find_clusters(points, i+1, restarts, iterations)
        mod_BIC = compute_BIC(points, model)
        print("mod BIC for run on ", i+1," clusters: ", mod_BIC)
        if mod_BIC < smallest_BIC:
            smallest_BIC = mod_BIC
            best_model_likelihood = likelihood
            best_model = model
            best_n = i+1
        else:
            #the value of the computed BIC is no longer decreasing, so break
            break
    print(f"Best value for n was {best_n}")
    return (best_model, best_model_likelihood)


def compute_BIC(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    error_total = 0
    #for d in dists:
    #    error_total += np.sqrt(np.square(d.var_x + d.var_y))
    n = len(points)
    k = len(dists)
    bic = -2*calc_log_likelihood(points, dists)+k*np.log(n)
   # bic = n*np.log(error_total)+k*np.log(n)
    return bic

def plot(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.scatter(x,y)
    plt.show()

def init_clusters(number=3, minN=0, maxN = 10):
    clusters = []
    for _ in range(number): 
        x = random.random() * (maxN - minN) + minN 
        y = random.random() * (maxN - minN) + minN 
        print(f"init: x {x}")
        print(f"init: y {y}")
        clusters.append(dist(x, 10, y, 10, 1 / number))
    return clusters

def plot_clusters(data, responsibility): 
    colors = ['red', 'green', 'blue', 'black', 'orange']
    labels = ['x', 'y', 'c']
    d = []

    for data, resp in zip(data, responsibility):
        most_resp = resp.argmax()
        d.append([*data, colors[most_resp]])


    df = pd.DataFrame.from_records(d, columns=labels)
    df.plot(kind='scatter', x='x', y='y', c=df.c)
    plt.show()


