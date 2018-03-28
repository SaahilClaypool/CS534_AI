#%% TESTING
from em import *

data = load_data()


dists, like = find_clusters(data, 3, restarts=3, iterations=100)

for d in dists: 
    print(d)

# plot(data)
plot_clusters(data, calc_responsibility(data, dists))
#6675.455617359649 w 4, or 6646?
#6663.29012876849 w 3
#6297.83420765347 w 2
#6972.853921251283 w 1

