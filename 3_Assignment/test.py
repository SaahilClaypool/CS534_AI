#%% TESTING
from em import *

data = load_data("small_sample.csv")

best_model, likeli, n = find_number_of_clusters(data, max_clusters=7, iterations=50)
plot_clusters(data, calc_responsibility(data, best_model))
