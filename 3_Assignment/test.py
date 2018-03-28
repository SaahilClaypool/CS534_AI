#%% TESTING
from em import *

data = load_data()


dists, like = find_clusters(data, 3, restarts=3, iterations=100)

print("BIC:",compute_BIC(data, dists))

for d in dists:
    print(d)

# plot(data)
plot_clusters(data, calc_responsibility(data, dists))
#new bics
#16242.941596706065 for 4
#16242.917838032536 for 3
#16437.26073170541 for 2