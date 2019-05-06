#%% prediction step

from numpy import dot
def kf_predict(theta, P, F, Q):
    theta = dot(F, theta)
    P = dot(F, P).dot(F.T) + Q
    return(theta,P)

#%% correlation step

from numpy import dot, sum, tile, linalg, identity
from numpy.linalg import inv

def kf_update(Theta, P, Y, X, R):
    IM = dot(X, Theta)
    IS = dot(X, P).dot(X.T) + R
    K = dot(P, X.T).dot(inv(IS))
    Theta = Theta + dot(K, (Y-IM))
    P = (identity(len(P)) - dot(K, X)).dot(P)

    return (Theta,P)

def kf_update_woodbury(Theta, P, Y, X, R):
    from src.data_sketching_kalman_filter.low_rank_approximation import spd_sys_approximate
    import numpy as np
    IM = dot(X, Theta)

    R_diag_inv = R.diagonal()**-1
    R_inv = np.diag(R_diag_inv)
    inv_IS_woodbury = R_inv - R_inv @ X @ np.linalg.inv(np.linalg.inv(P) + X.T @ R_inv @ X) @ X.T @ R_inv

    K = P @ X.T @ inv_IS_woodbury
    Theta = Theta + dot(K, (Y-IM))
    P_predict = (identity(len(P)) - dot(K, X)).dot(P)

    return (Theta,P_predict)

def kf_update_sketch_projection(Theta, P, Y, X, R, sketch_size = 0.2, iteration_step = 100):
    import numpy as np
    from src.data_sketching_kalman_filter.sketching_projection import sketching_projection

    IM = dot(X, Theta)
    IS = dot(X, P).dot(X.T) + R
    # K_test = dot(P, X.T).dot(inv(IS)).T

    B = np.dot(X, P)
    K = np.zeros((IS.shape[1], P.shape[0]))

    s_size = int(IS.shape[0] * sketch_size)
    for t in range(iteration_step):
        K = sketching_projection(K, IS, B, s_size)
        # print(np.linalg.norm(K-K_test, 'fro'))
    K = K.T

    Theta = Theta + np.dot(K, (Y-IM))
    P = (identity(len(P)) - dot(K, X)).dot(P)

    return (Theta,P)

if __name__ == '__main__':
    from src.data_sketching_kalman_filter.synthetic_data_generation import *
    from src.data_sketching_kalman_filter.kalman_filter import *
    from src.data_sketching_kalman_filter.utilis import *
    import numpy as np
    from time import time

    # some dimension setting
    D = 1000 # sample per time step
    p = 25 # num of feature
    sigma_w = 0.1 # state noise
    sigma_v = 1 # sample noise

    # predicted_theta0
    m0 = np.zeros((p, 1))
    m0[0] = 20
    m0[4] = -30
    theta_predict = np.zeros((p, 1))

    # predicted_initial_theta_covariance_matrix
    P_predict = np.identity(p)

    # dynamic system
    F = state_transition_matrix(p)
    theta = initial_state(p)
    Q = noise_cov_idp(p) * sigma_w ** 2
    R = noise_cov_idp(D) * sigma_v ** 2

    w = noise_generation(p, Q)
    v = noise_generation(D, R)
    theta = np.dot(F, theta) + w
    X = random_sample(p, D)
    y = np.dot(X, theta) + v

    theta_predict, P_predict = kf_predict(theta_predict, P_predict, F, Q)

    begin1 = time()
    theta_predict_1, P_predict_1 = kf_update_sketch_projection(theta_predict, P_predict, y, X, R, 0.025, 8)
    end1 = time()
    print('sketching method running time {}'.format(end1 - begin1))

    begin2 = time()
    theta_predict_2, P_predict_2 = kf_update(theta_predict, P_predict, y, X, R)
    end2 = time()
    print('inverse running time {}'.format(end2 - begin2))


