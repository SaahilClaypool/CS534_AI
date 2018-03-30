#%% TESTING
from em import *

data = load_data("sample5.csv")

# best_model, likeli, n = find_number_of_clusters(data, iterations=150, restarts=5)
best_model, p = find_clusters(data, number=5, iterations=150, restarts=6, tol=.0000000000000001)
print(best_model)
plot_clusters(data, calc_responsibility(data, best_model))
