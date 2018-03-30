#%% TESTING
from em import *

data = load_data("sample3.csv")

best_model, likeli, n = find_number_of_clusters(data, iterations=150, restarts=5)
# best_model, p = find_clusters(data, number=5, iterations=250, restarts=6, tol=.00000001, plot=True)
for iin, i in enumerate(best_model):
    print(f"model: {iin}, {i}")
plot_clusters(data, calc_responsibility(data, best_model))
