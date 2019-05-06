#%% generate the synthesis data

from src.data_sketching_kalman_filter.synthetic_data_generation import *
from src.data_sketching_kalman_filter.kalman_filter import *
from src.data_sketching_kalman_filter.utilis import *
import numpy as np
from time import time

def full_kf_mse(y, X, Q, R, F, theta_predict, P_predict,  approx_K = True, sketch_size = 0.2, iteration_step = 100):

    # full dimension kalman filter
    # kf_begin = time()
    # prediction step
    theta_predict, P_predict = kf_predict(theta_predict, P_predict, F, Q)
    # kf_end1 = time()
    # print('prediction step needs {} s'.format(kf_begin1 - kf_end1))
    # correlation step
    # kf_begin2 = time()
    if approx_K:
        theta_predict, P_predict = kf_update_sketch_projection(theta_predict, P_predict, y, X, R,  sketch_size, iteration_step)
    else:
        theta_predict, P_predict = kf_update(theta_predict, P_predict, y, X, R)
    # kf_end = time()
    # print('corrected step needs {} s'.format(kf_end2 - kf_begin2))
    # MSE.append(per_RSME(theta_predict, theta)**2)
    # if mute_print == False:
    #     print('mean sqaure error is {}'.format(per_RSME(theta_predict, theta)))
    # print('relative estimation error is {}'.format(relative_error(Theta_predict, theta, X, y)))
    # print('kalman_filiter processed finished, need total {}s'.format(kf_end-kf_begin))

    # return (sum(MSE)/N)**0.5
    return theta_predict, P_predict

if __name__ == "__main__":
    # some dimension setting
    N = 50 # time step
    D = 1500 # sample per time step
    p = 30 # num of feature
    sigma_w = 0.1 # state noise
    sigma_v = 1 # sample noise

    # predicted_theta0
    m0 = np.zeros((p, 1))
    m0[0] = 20
    m0[4] = -30

    theta = initial_state(p)

    # dynamic system
    F = state_transition_matrix(p)

    Q = noise_cov(p) * sigma_w ** 2
    R = noise_cov(D) * sigma_v ** 2

    _,  _, theta_set, X_set, y_set = synthetic_data_generate(N, p, Q, R, F, D, theta)

    running_time = []
    mse_set = []
    P_predict = np.identity(p)

    theta_predict = np.zeros((p, 1))
    for i in range(N):
        kf_begin = time()
        theta_predict, P_predict = full_kf_mse(y_set[i], X_set[i], Q, R, F, theta_predict, P_predict,
                                               approx_K = False, sketch_size = 0.05, iteration_step = 5)
        kf_end = time()
        rmse = per_RSME(theta_predict, theta_set[i])
        mse_set.append(rmse**2)
        running_time.append(kf_end-kf_begin)

    print('running time is {}'.format(np.mean(running_time)))
    print('mean sqaure error is {}'.format(np.mean(rmse)))

# sketch & projection works well with tall matrix X
# when the matrix is tall enough, sketch size would be just around the P/N is good enough

