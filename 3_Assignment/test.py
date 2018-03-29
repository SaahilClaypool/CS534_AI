#%% TESTING
from em import *

data = load_data("sample.csv")

# best_model, likeli, n = find_number_of_clusters(data, iterations=75, restarts=2)
best_model, p = find_clusters(data, number=3, iterations=150, restarts=4, tol=.0000000000000001)
print(best_model)
plot_clusters(data, calc_responsibility(data, best_model))
