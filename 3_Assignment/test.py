#%% TESTING
from em import *

data = load_data()


# data = [
#     (-1,0), (1, 0), (0, 1), (0,-1),
#     (9,10), (11, 10), (10, 9), (10,11),
#     ]
# dists = [dist(0,10,0,10,.5), dist(10,10,10,10,.5), dist(0,10,10,10,.5)]
dists = init_clusters(3)

resp = calc_responsibility(data, dists)
for i in range(70):
    dists = update_dists(data, dists)

for d in dists: 
    print(d)

# plot(data)
plot_clusters(data, calc_responsibility(data, dists))