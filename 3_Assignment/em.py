"""
Perform EM on a given input file and finds 3 clusters from the data
"""

#%%
import csv

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

#%% TESTING
data = load_data()

import matplotlib.pyplot as plt
def plot(data):
    x = [d[0] for d in data]
    y = [d[1] for d in data]
    plt.scatter(x,y)

plot(data)
