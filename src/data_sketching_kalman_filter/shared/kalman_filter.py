#%% prediction step

def kf_predict(theta, P, F, Q):
    theta = dot(F, theta)
    P = dot(F, P).dot(F.T) + Q
    return(theta,P)

#%% correlation step

from numpy import dot, identity
from numpy.linalg import inv

def kf_update(Theta, P, Y, X, R, plot_k = False):
    IM = dot(X, Theta)
    IS = dot(X, P).dot(X.T) + R
    K = dot(P, X.T).dot(inv(IS))
    Theta = Theta + dot(K, (Y-IM))
    P = (identity(len(P)) - dot(K, X)).dot(P)
    if plot_k:
        return K
    else:
        return Theta,P

def kf_update_woodbury(Theta, P, Y, X, R):
    import numpy as np
    IM = dot(X, Theta)

    R_diag_inv = R.diagonal()**-1
    R_inv = np.diag(R_diag_inv)
    inv_IS_woodbury = R_inv - R_inv @ X @ np.linalg.inv(np.linalg.inv(P) + X.T @ R_inv @ X) @ X.T @ R_inv

    K = P @ X.T @ inv_IS_woodbury
    Theta = Theta + dot(K, (Y-IM))
    P_predict = (identity(len(P)) - dot(K, X)).dot(P)

    return Theta,P_predict

def kf_update_sketch_projection(Theta, P, Y, X, R, sketch_size = 0.2, iteration_step = 100, plot_k = False):
    import numpy as np
    from src.data_sketching_kalman_filter.shared.sketching_projection import sketching_projection

    IM = dot(X, Theta)
    IS = dot(X, P).dot(X.T) + R
    # K_test = dot(P, X.T).dot(inv(IS)).T

    B = np.dot(X, P)
    K = np.zeros((IS.shape[1], P.shape[0]))
    K_set = []
    K_set.append(K)
    s_size = int(IS.shape[0] * sketch_size)
    for t in range(iteration_step):
        K = sketching_projection(K, IS, B, s_size)
        K_set.append(K)
        # print(np.linalg.norm(K-K_test, 'fro'))
    K = K.T

    Theta = Theta + np.dot(K, (Y-IM))
    P = (identity(len(P)) - dot(K, X)).dot(P)
    if plot_k:
        return K_set
    else:
        return Theta, P

if __name__ == '__main__':
    from src.data_sketching_kalman_filter.shared.synthetic_data_generation import *
    import numpy as np

    # some dimension setting
    D = 500 # sample per time step
    p = 50 # num of feature
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

    K_set = kf_update_sketch_projection(theta_predict, P_predict, y, X, R, 0.05, 20, plot_k=True)

    K = kf_update(theta_predict, P_predict, y, X, R, plot_k=True)

    error1 = round(np.linalg.norm(K - K_set[1].T, 'fro'),4)
    error10 = round(np.linalg.norm(K - K_set[10].T, 'fro'),4)
    error20 = round(np.linalg.norm(K - K_set[20].T, 'fro'),4)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 1)
    axs[0].imshow(K, cmap=plt.cm.BuPu_r)
    axs[0].set_xlim(0, 300)
    axs[0].set_title('true solution')
    axs[1].imshow(K_set[1].T, cmap=plt.cm.BuPu_r)
    axs[1].set_xlim(0, 300)
    axs[1].set_title('1 iteration')
    axs[2].imshow(K_set[5].T, cmap=plt.cm.BuPu_r)
    axs[2].set_xlim(0, 300)
    axs[2].set_title('10 iteration')
    axs[3].imshow(K_set[10].T, cmap=plt.cm.BuPu_r)
    axs[3].set_xlim(0, 300)
    axs[3].set_title('20 iteration')
    plt.tight_layout()
    plt.savefig('sketch.png')
    plt.show()




    # inv_IS = inv(IS)
    # v, d, u = np.linalg.svd(inv_IS)
    # d[d<0.1] = 0
    # inv_IS_test = (v @ np.diag(d) @ u)
    #
    # K_test = dot(P, X.T).dot(inv_IS_test)
    # K_true = dot(P, X.T).dot(inv_IS)
    #
    #
    # fig, axs = plt.subplots(2, 1)
    # axs[0].imshow(K_true, cmap=plt.cm.BuPu_r)
    # axs[0].set_xlim(0, 100)
    # axs[0].set_title('true Kalman gain')
    # axs[1].imshow(K_test, cmap=plt.cm.BuPu_r)
    # axs[1].set_xlim(0, 100)
    # axs[1].set_title('approx Kalman gain')
    # plt.tight_layout()
    # plt.savefig('K_compare.png')
    # plt.show()



