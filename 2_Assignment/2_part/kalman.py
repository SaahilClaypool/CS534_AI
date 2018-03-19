from numpy.linalg import *
import numpy as np

x_start = np.matrix([[1000], [10]])
transition_eq = np.matrix(
              [[1, 1], #update by increasing gdp by velocity
              [0, 1]]) # vel stays the same

sensor_input = np.matrix(
                [[3], # immigrants
                [12]]) # exports
                    
sensor_eq = np.matrix(
            [[.002, 0], # 2 immigrants per 1000 gdp
             [.01, 0]]) # 1 export per 100 gdp

x_predict = transition_eq * x_start
z_predict = sensor_eq * x_predict
print("z_pred", z_predict)

residual = sensor_input - z_predict

print(residual)

# calc gain


x_cov = np.matrix([[1, 0], # must be 4x4
                       [.1,.1]])
x_cov = np.dot(np.dot(transition_eq, x_cov),  transition_eq.T)

sensor_cov = np.dot(np.dot(transition_eq, x_cov),  transition_eq.T)

k_gain = multi_dot([x_cov, sensor_eq.T, inv(sensor_cov)])

print(sensor_cov)
print(k_gain)

final_x = x_predict + k_gain.dot(residual)
print("predicted", x_predict)
print("residual", residual)
print("final", final_x)
print(final_x)




