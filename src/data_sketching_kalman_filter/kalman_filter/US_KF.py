from src.data_sketching_kalman_filter.shared.synthetic_data_generation import *
from src.data_sketching_kalman_filter.shared.utilis import *
from src.data_sketching_kalman_filter.shared.sketching import *
import numpy as np
from time import time
from src.data_sketching_kalman_filter.shared.kalman_filter import *

def us_kf_mse(N, D, p, Q, R, F, theta, theta_predict, P_predict, tau, mute_print = False):
    MSE = []
    d_set = []
    for t in range(N):
        begin = time()
        w = noise_generation(p,  Q)
        v = noise_generation(D,  R)
        theta = np.dot(F, theta) + w
        X = random_sample(p, D)
        y = np.dot(X, theta) + v
        end = time()

        # prediction step
        theta_predict, P_predict = kf_predict(theta_predict, P_predict, F, Q)

        # Correction Step
        d = 0

        for i in range(D):
            true_gamma = np.dot(X[i, :], np.dot(P_predict, X[i,:]))
            s = generate_s(X[i, :], P_predict, R[i,i])
            k_n = generate_kn(P_predict, X[i].reshape((-1,1)), s)
            k_n = k_n.reshape((-1, 1))
            e = error(y[i,0], X[i], theta_predict)[0]
            e_line = e * s**(-0.5)
            D_pp = 0.5 * e_line **2 * (2 * true_gamma + (true_gamma**2)/R[i,i]) * 1/s

            #update theta and P
            if D_pp >= tau /(i+1):
                d = d+1
                theta_predict = theta_predict + k_n*e
                P_predict = update_predict_p(P_predict, X[i].reshape((-1, 1)), s)
            else:
                mu = generate_mu(X[i].reshape((-1, 1)), true_gamma, R[i,i])
                theta_predict = theta_predict + mu*X[i].reshape((-1, 1))*e
        d_set.append(d)

        MSE.append(per_RSME(theta_predict, theta) ** 2)
        if mute_print == False:
            print('mean sqaure error is {}, redcued dimension is {}'.format(per_RSME(theta_predict, theta), d))

    return ((sum(MSE)/N)**0.5, np.mean(d_set))

if __name__ == "__main__":
    # some dimension setting
    N = 100
    D = 500
    p = 50
    sigma_w = 0.1
    sigma_v = 1

    # sketching parameter
    tau = 100

    # predicted_theta0
    m0 = np.zeros((p, 1))
    m0[0] = 20
    m0[4] = -30
    theta_predict = m0

    # predicted_initial_theta_covariance_matrix
    P_predict = 0.04 * np.identity(p)

    # dynamic system
    F = state_transition_matrix(p)
    theta = initial_state(p)
    Q = noise_cov(p) * sigma_w ** 2
    R = noise_cov(D) * sigma_v ** 2

    us_kf_mse(N, D, p, Q, R, F, theta, theta_predict, P_predict, tau)