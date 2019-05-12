from src.data_sketching_kalman_filter.kalman_filter.full_KF import *
from src.data_sketching_kalman_filter.kalman_filter.RS_KF import *
from src.data_sketching_kalman_filter.kalman_filter.CoutingSketch_KF import *
from src.data_sketching_kalman_filter.kalman_filter.AC_sketching_KF import *

# Number of time slot
N = 50

# Number of observation per time slot
# for the convinence of JL transform >_<
D = 1500

# Number of features
p = 50

# scaler for covariance matrix of noise vector
sigma_w = 0.1
sigma_v = 1

# initial state vector
m0 = np.zeros((p, 1))
m0[0] = 20
m0[4] = -30

# transition matrix
F = state_transition_matrix(p)

# add the initial state with some noise
theta = initial_state(p)

# covariance matrix for white noise of state
Q = noise_cov(p) * sigma_w ** 2

# covariance matrix for white noise of observation
R = noise_cov_idp(D) * sigma_v ** 2

_,  _, theta_set, X_set, y_set = synthetic_data_generate(N, p, Q, R, F, D, theta)


d_per_set = [0.025, 0.05, 0.10, 0.15, 0.2, 0.25]

rs_reuslt_dict = {}
for d_per in d_per_set:
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
            theta_predict, P_predict = rs_kf_mse(y_set[i], X_set[i], Q, R, F, theta_predict, P_predict, d,
                                                 approx_K=False, sketch_size=0.10, iteration_step=4)
            kf_end = time()
            rmse = per_RSME(theta_predict, theta_set[i])
            mse_set.append(rmse ** 2)
            running_time.append(kf_end - kf_begin)

        conclusion_set_rt.append(np.mean(running_time))
        conclusion_set_rmse.append(np.mean(rmse))

    rs_reuslt_dict[d_per] = np.mean(conclusion_set_rmse)
    print('mse is:',  np.mean(conclusion_set_rmse), 'with', 'd/D to be:', d_per)

ck_reuslt_dict = {}
for d_per in d_per_set:
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
            theta_predict, P_predict = count_kf_mse(y_set[i], X_set[i], Q, R, F, theta_predict, P_predict, d,
                                                 approx_K=False, sketch_size=0.10, iteration_step=4)
            kf_end = time()
            rmse = per_RSME(theta_predict, theta_set[i])
            mse_set.append(rmse ** 2)
            running_time.append(kf_end - kf_begin)

        conclusion_set_rt.append(np.mean(running_time))
        conclusion_set_rmse.append(np.mean(rmse))

    ck_reuslt_dict[d_per] = np.mean(conclusion_set_rmse)
    print('mse is:',  np.mean(conclusion_set_rmse), 'with', 'd/D to be:', d_per)

ac_tau_set = [3.5, 2.8, 2.5, 2.2, 2, 1.8, 1.5, 1.2]
mu = 0.001
ac_reuslt_dict = {}
for tau in ac_tau_set:

    conclusion_set_rt = []
    conclusion_set_rmse = []
    running_time = []
    for mm in range(10):

        mse_set = []
        len_s_set = []
        P_predict = np.identity(p)
        theta_predict = np.zeros((p, 1))
        for i in range(N):
            kf_begin = time()
            theta_predict, P_predict, len_s = ac_kf_mse(y_set[i], X_set[i], Q, R, F, tau, mu, theta_predict, P_predict,
                                                   approx_K = False, sketch_size = 0.25, iteration_step = 4)
            kf_end = time()
            rmse = per_RSME(theta_predict, theta_set[i])
            mse_set.append(rmse ** 2)
            running_time.append(kf_end - kf_begin)
            len_s_set.append(len_s)

        conclusion_set_rmse.append(np.mean(rmse))

    d_per = np.mean(len_s_set) / D

    ac_reuslt_dict[d_per] = np.mean(conclusion_set_rmse)
    print('mse is:',  np.mean(conclusion_set_rmse), 'with', 'd/D to be:', np.mean(len_s_set)/D, 'and running time to be:', np.mean(running_time))

kp_reuslt_dict = {}
for d_per in d_per_set:
    conclusion_set_rt = []
    conclusion_set_rmse = []
    for mm in range(2):
        running_time = []
        mse_set = []
        P_predict = np.identity(p)
        theta_predict = np.zeros((p, 1))
        for i in range(N):
            kf_begin = time()
            theta_predict, P_predict = full_kf_mse(y_set[i], X_set[i], Q, R, F, theta_predict, P_predict,
                                                 approx_K=True, sketch_size=d_per, iteration_step=5)
            kf_end = time()
            rmse = per_RSME(theta_predict, theta_set[i])
            mse_set.append(rmse ** 2)
            running_time.append(kf_end - kf_begin)

        conclusion_set_rt.append(np.mean(running_time))
        conclusion_set_rmse.append(np.mean(rmse))

    kp_reuslt_dict[d_per] = np.mean(conclusion_set_rmse)
    print('mse is:',  np.mean(conclusion_set_rmse), 'with', 'd/D to be:', d_per)


rs_sketching_d = list(rs_reuslt_dict.keys())
rs_sketching_rmse = list(rs_reuslt_dict.values())

ck_sketching_d = list(ck_reuslt_dict.keys())
ck_sketching_rmse = list(ck_reuslt_dict.values())

ac_sketching_d = list(ac_reuslt_dict.keys())
ac_sketching_rmse = list(ac_reuslt_dict.values())

kp_sketching_d = list(kp_reuslt_dict.keys())
kp_sketching_rmse = list(kp_reuslt_dict.values())

import matplotlib.pyplot as plt
plt.plot(ck_sketching_d, ck_sketching_rmse,  marker='o', markerfacecolor='blue', markersize=12, color='skyblue',
         linewidth=4, label = 'Count Sketch sketching')
plt.plot(ac_sketching_d, ac_sketching_rmse, marker='x', color='red', linewidth=2, label = 'Censoring_based_sketching')
plt.plot(kp_sketching_d, kp_sketching_rmse, marker='*', color='orange', linewidth=2, label = 'sketching & Projection')
plt.plot(rs_sketching_d, rs_sketching_rmse, marker='_', color='black', linewidth=2, label = 'Random Sampling sketching')
plt.xlabel('d/D')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('rmse_tendency.png')
plt.show()
# notice that the RMSE for full data kalman filter is around 0.085



running_time = []
mse_set = []
P_predict = np.identity(p)
theta_predict = np.zeros((p, 1))
for i in range(N):
    kf_begin = time()
    theta_predict, P_predict = full_kf_mse(y_set[i], X_set[i], Q, R, F, theta_predict, P_predict,
                                           approx_K=False, sketch_size=d_per, iteration_step=5)
    kf_end = time()
    rmse = per_RSME(theta_predict, theta_set[i])
    mse_set.append(rmse ** 2)
    running_time.append(kf_end - kf_begin)

conclusion_set_rt.append(np.mean(running_time))
conclusion_set_rmse.append(np.mean(rmse))

kp_reuslt_dict[d_per] = np.mean(conclusion_set_rmse)
print('mse is:', np.mean(conclusion_set_rmse), 'with', 'd/D to be:', d_per)


# apply both count sketch and projection & sketch method
import pandas as pd
import seaborn as sns

sketch_size = [0.05, 0.1, 0.15, 0.2, 0.25]
d_D = [0.3, 0.35, 0.40, 0.45, 0.5]

Index = [str(i) for i in sketch_size]
Cols = [str(i) for i in d_D]
df = pd.DataFrame(index = Index, columns = Cols)

for d_per in d_D:
    for s in sketch_size:
        d = int(d_per * D)
        running_time = []
        mse_set = []
        P_predict = np.identity(p)
        theta_predict = np.zeros((p, 1))
        for i in range(N):
            kf_begin = time()
            theta_predict, P_predict = count_kf_mse(y_set[i], X_set[i], Q, R, F, theta_predict, P_predict, d,
                                                 approx_K=True, sketch_size=s, iteration_step=5)
            kf_end = time()
            rmse = per_RSME(theta_predict, theta_set[i])

            running_time.append(kf_end - kf_begin)
            mse_set.append(rmse)
        mean_run_time = np.mean(running_time)
        mean_rmse = np.mean(mse_set)
        df.loc[str(s), str(d_per)] = mean_rmse
        print('average rmse is {} for d_per = {} and sketch size = {}, running time = {}'.format(mean_rmse, d_per, s, mean_run_time))

sns.heatmap(df, annot=True)
plt.xlabel('data sketch size')
plt.ylabel('projection sketch size')
plt.savefig('skech_sketch.png')
plt.show()


