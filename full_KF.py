#%% generate the synthesis data

from synthetic_data_generation import *
from kalman_filter import *
from utilis import *
import numpy as np
from time import time

# some dimension setting
N = 100
D = 1000
p = 50
sigma_w = 0.1
sigma_v = 1

# predicted_theta0
m0 = np.zeros((p,1))
m0[0] = 20
m0[4] = -30
theta_predict = m0

# predicted_initial_theta_covariance_matrix
P_predict = 0.04*np.identity(p)


# dynamic system
F = state_transition_matrix(p)
theta = initial_state(p)
Q = noise_cov(p)
R = noise_cov(D)

MSE = []
for i in range(N):
    begin = time()
    w = noise_generation(p, sigma_w, Q)
    v = noise_generation(D, sigma_v, R)
    theta = np.dot(F, theta) + w
    X = random_sample(p, D)
    y = np.dot(X, theta) + v
    end = time()
    # print('time slot {} has passed, need total {}s'.format(i, end-begin))

# full dimension kalman filter
# for i in range(N):
    kf_begin = time()
    # prediction step
    theta_predict, P_predict = kf_predict(theta_predict, P_predict, F, Q)
    # correlation step
    theta_predict, P_predict = kf_update(theta_predict, P_predict, y, X, R)
    kf_end = time()
    MSE.append(per_RSME(theta_predict, theta)**2)
    print('relative estimation error is {}'.format(per_RSME(theta_predict, theta)))
    # print('relative estimation error is {}'.format(relative_error(Theta_predict, theta, X, y)))
    # print('kalman_filiter processed finished, need total {}s'.format(kf_end-kf_begin))

(sum(MSE)/N)**0.5