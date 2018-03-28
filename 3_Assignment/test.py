#%% TESTING
from em import *

data = load_data("custom_sample.csv")

best_model, likeli, n = find_number_of_clusters(data, iterations=75, restarts=2)
plot_clusters(data, calc_responsibility(data, best_model))
