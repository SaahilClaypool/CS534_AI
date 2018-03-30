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

import sys


class dist(object):
    """
    the Dist class will represent a model of the gaussian distribution that we will for our EM
    """
    def __init__(self, mean_x, var_x, mean_y, var_y, fraction_of_total):
        self.mean_x = mean_x
        self.mean_y = mean_y

        self.var_x = var_x
        self.var_y = var_y

        self.fraction_of_total = fraction_of_total

        #set up covariance matrix and the multivariate normal distribution
        self.cov = [[self.var_x, 0], [0, self.var_y]]
        self.mvn = stats.multivariate_normal([self.mean_x, self.mean_y],cov=self.cov)
    
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
        # Note: prob is really small, but shouldn't matter after you normalize
        return self.fraction_of_total * self.mvn.pdf(coord)

    pass

def load_data(filename="sample.csv"): 
    """
    return a list of [(x,y)]
    """
    coords = np.empty([0,2])
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            coords = np.append(coords, [[float(row[0]), float(row[1])]], axis=0)
    return coords

def calc_responsibility(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    """
    args: 
        points: matrix for [X,Y] each row is point
        dists: a list of dist objects, i.e. our model
    
    returns:    
        matrix of R x C
        where R is each point, C is each distribution
    """
    total_probs = []
    for point in points: 
        probs = []
        for c in dists:
            r = c.responsibility(point)
            probs.append(r)
        #normalalize
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
    #for each point in our data...
    for pi, point in enumerate(points):
        p_sum = 0
        #get the responsibility of this point for each distribution
        for ci, c in enumerate(dists):
            r = c.responsibility(point)
            p_sum += r
        #add the log of the sum of the responsibilities from each distribution for this datum
        log_like += np.log(p_sum)
    return log_like

def update_dists(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    """
    Update the location of the given distributions based on their 'responsibility' 
    for each point
    """
    #calculate all the responsibilities for each point and each distribution
    resp = calc_responsibility(points, dists)
    updated_dists = []
    for c, d in enumerate(dists):
        # total amount of responsibility for this cluster
        mc = max(.1, np.sum(resp[:, c]))
        # normalize by number of points
        fraction_of_total = mc / len(points)

        new_mean_x = 0.1
        new_mean_y = 0.1
        #update the x and y coordinates for this distribution based on its responsibility for each point
        for r, point in enumerate(points):
            new_mean_x += resp[r][c] * point[0]
            new_mean_y += resp[r][c] * point[1]
        new_mean_x = new_mean_x / mc
        new_mean_y = new_mean_y / mc

        new_var_x = 0.1
        new_var_y = 0.1
        #update the x and y variances
        for r, point in enumerate(points): 
            new_var_x += resp[r][c]  * (point[0] - new_mean_x)**2
            new_var_y += resp[r][c]  * (point[1] - new_mean_y)**2
        new_var_x = new_var_x / mc
        new_var_y = new_var_y / mc

        #add updated dist to new list
        new_dist = dist(new_mean_x , new_var_x , new_mean_y , new_var_y , fraction_of_total)
        updated_dists.append(new_dist)

    return updated_dists

def find_clusters(points: Sequence[Tuple[int, int]], number: int, restarts: int = 0, iterations = 75, tol: float = .00001, plot=False):

    """
    Find the best model with the given number of clusters and restarts
    """
    if (restarts < 1 or plot): restarts = 1
    best_model: Sequence[dist]
    prev_likelihood = -math.inf
    best_likelihood = -math.inf

    for r in range(restarts):
        # initialize initial clusters
        dists = init_clusters(number, data=points)
        llh = []

        # wrap the operations in this for loop so that we are guarenteed to stop in a reasonable amount of time
        for i in range(iterations):
            # update dists and calculate log likelihood of the model
            dists = update_dists(points, dists)
            new_likelihood = calc_log_likelihood(points, dists)
            #if change in likelihood is below tolerance, then break
            # print(f"diff {math.fabs(prev_likelihood - new_likelihood)}")
            llh.append(new_likelihood)
            if i == 60:
                if(plot):
                    plt.plot(llh)
                    plt.show()
                    print(f"Iterations: {i}")
                    for cluster_num, c in enumerate(dists):
                        print(f"{cluster_num}: {c}")

            # if change in likelihood from previous computation is below tolerance, then break
            if math.fabs(prev_likelihood - new_likelihood) < tol:
                break
            else:
                prev_likelihood = new_likelihood

        # if the log likelihood of this new model is better than previous one, replace
        likeli = calc_log_likelihood(points, dists)
        if (likeli >= best_likelihood):
            best_likelihood = likeli
            best_model = dists
        
    if(plot):
        plt.plot(llh)
        plt.show()
        print(f"Iterations: {i}")
        for cluster_num, c in enumerate(dists):
            print(f"{cluster_num}: {c}")

    return best_model, best_likelihood

def find_number_of_clusters(points: Sequence[Tuple[int, int]], restarts: int = 0, iterations = 75, max_clusters=15, tol: float = .0000000000000001) -> Tuple[Sequence[dist], float]:
    """
    Find the best model by determining the best number of clusters
    """
    # default restarts is 1
    if (restarts < 1): restarts = 1
    smallest_BIC = math.inf
    best_model: Sequence[dist]
    best_model_likelihood = 0
    best_n = 0

    max_clusters = min(len(points), max_clusters)

    # our model should have, at most, the same number of clusters as points.
    for i in range(2, max_clusters):
        # find the best model for the current number of clusters we are trying
        model, likelihood = find_clusters(points, i, restarts, iterations, tol=tol)

        # compute the BIC value of the best model for this number of clusters
        mod_BIC = compute_BIC(points, model)
        print("model BIC calculated for run on ", i," clusters: ", mod_BIC)

        # if the BIC for this number of clusters is smaller than the previous, replace the best model/cluster choices
        if mod_BIC < smallest_BIC:
            smallest_BIC = mod_BIC
            best_model_likelihood = likelihood
            best_model = model
            best_n = i
        else:
            #the value of the computed BIC is no longer decreasing, so break
            break

    print(f"Best value for n was {best_n}")
    return (best_model, best_model_likelihood, best_n)


def compute_BIC(points: Sequence[Tuple[int, int]], dists: Sequence[dist]):
    """
    Given a model and data, compute the bayesian information criterion.
    We will choose the "best" model / number of clusters by looking for the minimum value of BIC
    """
    n = len(points)
    # set the number of parameters as 3 times the number of clusters used
    # consider the parameters as mean_x, mean_y, and covariance
    k = 3*len(dists)

    bic = -2*calc_log_likelihood(points, dists)+k*np.log(n)
    return bic


def plot(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.scatter(x, y)
    plt.show()


def init_clusters(number=3, minN=-1, maxN = -1, data=[]):
    """
    Initialize clusters with default x/y values and variance.
    The initialization point will be random, between the minimum and maximum points in our data
    """

    minx = miny = minN
    maxx = maxy = maxN
    varx = vary = 10
    if len(data) is not 0:
        minx = np.min(np.array(data)[:,0])
        maxx = np.max(np.array(data)[:,0])
        miny = np.min(np.array(data)[:,1])
        maxy = np.max(np.array(data)[:,1])
        varx = np.var(np.array(data)[:,0]) / number
        vary = np.var(np.array(data)[:,1]) / number
    clusters = []
    for _ in range(number): 
        x = random.random() * (minx - maxx) + minx
        y = random.random() * (miny - maxy) + maxy
        clusters.append(dist(x, varx, y, vary, 1 / number))

    return clusters


def plot_clusters(data, responsibility):
    """
    Plot out clusters and color points to match their most probable cluster.
    """
    colors = ['red', 'green', 'blue', 'black', 'orange', 'steelblue', 'cyan', 'pink', 'purple','steelblue']
    labels = ['x', 'y', 'c']
    d = []

    for data, resp in zip(data, responsibility):
        most_resp = resp.argmax()
        d.append([*data, colors[most_resp]])

    df = pd.DataFrame.from_records(d, columns=labels)
    df.plot(kind='scatter', x='x', y='y', c=df.c, alpha=.5)
    plt.show()

def main():
    """
    take two inputs:
        - a file containing data points
        - the number of clusters to use (or the letter X to find the best number of clusters for the data)
    """
    data_file = sys.argv[1]
    data = load_data(data_file)
    print("Data loaded")
    if sys.argv[2] == "X":
        print("Beginning computation EM and determining best number of clusters based on BIC:")
        best_model, likeli, num = find_number_of_clusters(data, iterations=75, restarts=10)
        print("Best number of clusters: ", num)
    else:
        num = int(sys.argv[2])
        print("Beginning computation of EM on ", num, " clusters:")
        best_model, likeli = find_clusters(data, number=num,  iterations=75, restarts=10)
    print(" ")
    print("Final calculated log likelihood of the model: ", likeli)
    print("Final model parameters:")
    for i in range(len(best_model)):
        d = best_model[i]
        print("Distribution #", i+1)
        print(f"Distribution center: ({d.mean_x}, {d.mean_y}), X Variance: {d.var_x}, Y Variance: {d.var_y}")
        print(f"Fraction of data points that this distribution is most responsible for: {d.fraction_of_total}")
    if num <= 10:
        plot_clusters(data, calc_responsibility(data, best_model))
    else:
        print("Number of clusters too large to plot with pretty colors.")
    print("Final calculated log likelihood of the model: ", likeli)

if __name__ == "__main__":
    main()