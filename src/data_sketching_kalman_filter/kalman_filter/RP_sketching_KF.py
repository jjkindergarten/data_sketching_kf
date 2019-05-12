from src.data_sketching_kalman_filter.shared.synthetic_data_generation import *
from src.data_sketching_kalman_filter.shared.sketching import *
import numpy as np
from src.data_sketching_kalman_filter.shared.utilis import *
from time import time
from src.data_sketching_kalman_filter.shared.kalman_filter import *

def rp_kf_mse(y, X, Q, R, F, theta_predict, P_predict, d, approx_K = True, sketch_size = 0.2, iteration_step = 100):

    theta_predict, P_predict = kf_predict(theta_predict, P_predict, F, Q)
    # data reduction
    X_sk, y_sk, R_sk = fjl_sketching(X, y, R, d)

    if approx_K:
        theta_predict, P_predict = kf_update_sketch_projection(theta_predict, P_predict, y_sk, X_sk, R_sk,  sketch_size, iteration_step)
    else:
        theta_predict, P_predict = kf_update(theta_predict, P_predict, y_sk, X_sk, R_sk)

    return theta_predict, P_predict

if __name__ == "__main__":
    # some dimension setting
    N = 50 # time step
    D = 1500 # sample per time step
    p = 25 # num of feature
    d_per = 0.4
    d = int(d_per * D)
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
    R_get_data = noise_cov(D) * sigma_v ** 2

    _,  _, theta_set, X_set, y_set = synthetic_data_generate(N, p, Q, R_get_data, F, D, theta, detect_2power = True)

    R = noise_cov(D, detect_2power=True) * sigma_v ** 2

    d_per = 0.15
    d = int(d_per * D)

    conclusion_set_rt = []
    conclusion_set_rmse = []
    for mm in range(10):

        running_time = []
        mse_set = []
        P_predict = np.identity(p)
        theta_predict = np.zeros((p, 1))
        for i in range(N):
            kf_begin = time()
            theta_predict, P_predict = rp_kf_mse(y_set[i], X_set[i], Q, R, F, theta_predict, P_predict, d,
                                                   approx_K = False, sketch_size = 0.08, iteration_step = 4)
            kf_end = time()
            rmse = per_RSME(theta_predict, theta_set[i])
            mse_set.append(rmse**2)
            running_time.append(kf_end-kf_begin)

        print('running time is {}'.format(np.mean(running_time)))
        print('mean sqaure error is {}'.format(np.mean(rmse)))

        conclusion_set_rt.append(np.mean(running_time))
        conclusion_set_rmse.append(np.mean(rmse))

    np.mean(conclusion_set_rt)
