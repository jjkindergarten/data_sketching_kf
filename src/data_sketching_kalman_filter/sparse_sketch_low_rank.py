from src.data_sketching_kalman_filter.synthetic_data_generation import *
from src.data_sketching_kalman_filter.kalman_filter import *
from src.data_sketching_kalman_filter.sketching import *
import numpy as np
from src.data_sketching_kalman_filter.utilis import *
from time import time


def count_kf_mse(N, D, p, Q, R, F, theta, Theta_predict, P_predict, d, mute_print = False, ignore_first_n_step = 10):
    MSE = []
    for i in range(N):
        begin = time()
        w = noise_generation(p, Q)
        v= noise_generation(D, R)
        theta = np.dot(F, theta) + w
        X = random_sample(p, D)
        y = np.dot(X, theta) + v
        end = time()
        # print('time slot {} has passed, need total {}s'.format(i, end-begin))

    # full dimension kalman filter
    # for i in range(N):
        kf_begin = time()
        # prediction step
        Theta_predict, P_predict = kf_predict(Theta_predict, P_predict, F, Q)

        # data reduction
        X_sk, y_sk, R_sk = count_sketching(X, y, R, d)

        # correlation step
        Theta_predict, P_predict = kf_update(Theta_predict, P_predict, y_sk, X_sk, R_sk)
        kf_end = time()
        MSE.append(per_RSME(Theta_predict, theta)**2)
        if mute_print == False:
            print('relative estimation error is {}'.format(per_RSME(Theta_predict, theta)))
        # print('relative estimation error is {}'.format(relative_error(Theta_predict, theta, X, y)))
        # print('kalman_filiter processed finished, need total {}s'.format(kf_end-kf_begin))

    MSE = MSE[ignore_first_n_step:]
    return (sum(MSE)/len(MSE))**0.5

if __name__ == "__main__":
    # some dimension setting
    N = 110
    D = 512
    p = 50
    sigma_w = 0.01
    sigma_v = 1

    # sketching size
    d = 50

    # predicted_theta0
    m0 = np.zeros((p, 1))
    m0[0] = 20
    m0[4] = -30
    Theta_predict = np.zeros((p, 1))

    # predicted_initial_theta_covariance_matrix
    P_predict = np.identity(p)

    # dynamic system
    F = state_transition_matrix(p)
    theta = initial_state(p)
    Q = noise_cov(p) * sigma_w**2
    R = noise_cov(D) * sigma_v**2

    count_kf_mse(N, D, p, Q, R, F, theta, Theta_predict, P_predict, d, ignore_first_n_step = 10)
