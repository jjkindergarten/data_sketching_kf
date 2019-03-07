from synthetic_data_generation import *
from kalman_filter import *
from utilis import *
from sketching import *
import numpy as np
from utilis import *
from time import time
from numpy.linalg import norm

# some dimension setting
N = 100
D = 1024
p = 50
sigma_w = 0.01
sigma_v = 1

#sketching parameter
k = 1
tau = 0.01

# predicted_theta0
m0 = np.zeros((p,1))
m0[0] = 20
m0[4] = -30
Theta_predict = m0

# predicted_initial_theta_covariance_matrix
P_predict = 0.04*np.identity(p)


# dynamic system
F = state_transition_matrix(p)
theta = initial_state(p)

MSE = []
for i in range(N):
    begin = time()
    w, Q = noise_generation(p, sigma_w)
    v, R = noise_generation(D, sigma_v)
    theta = np.dot(F, theta) + w
    X = random_sample(p, D)
    y = np.dot(X, theta) + v
    end = time()

    # prediction step
    Theta_predict, P_predict = kf_predict(Theta_predict, P_predict, F, Q)

    # Correction Step
    for i in range(D):
        true_gamma = 1/p*norm(X[i:,])**2*np.trace(P_predict)
        s = generate_s(X[i, :], P_predict, R[i,i])
        k_n = generate_kn(P_predict, X[i,:], s)
        e = error(y[i], X[i, :], Theta_predict)
        e_line = e * s**-0.5
        D_pp = 0.5 * e_line **2 * (2 * true_gamma + (true_gamma/sigma_v[i,i])**2) * 1/s

        #update theta and P
        if D_pp >= tau /i:
            Theta_predict = Theta_predict + k_n*e
            P_predict = update_predict_p(P_predict, X[i,:], s)
        else:
            mu = generate_mu(X[i,:], true_gamma, R[i,i])
            Theta_predict = Theta_predict + mu*X[:,i]*e

