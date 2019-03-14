full_kf_RMSE = 0.165

#rp_sketching, d/D = 0.25, RMSE = 0.42; d/D = 0.1, RMSE = 0.86; d/D = 0.15, RMSE = 0.60; d/D =0.05, RMSE = 1.59
#rp_sketching, d/D = 0.3, RMSE = 0.36, d/D = 0.4, RMSE = 0.30; d/D = 0.5, RMSE = 0.267; d/D = 0.6, RMSE = 0.23

#AC_sketching, d/D = 0.5, RMSE = 0.27; d/D = 0.45, RMSE = 0.275; d/D = 0.235, RMSE = 0.39; d/D = 0.1, RMSE = 1.15
#AC_sketching, d/D = 0.2, RMSE = 0.61; d/D = 0.52, RMSE = 0.20

#US_sketching, d/D = 0.87, RMSE = 0.58; d/D = 0.137, RMSE = 0.40; d/D = 0.195, RMSE = 0.319; d/D = 0.23, RMSE = 0.30; d/D = 0.3, RMSE = 0.27;
# d/D = 0.39, RMSE = 0.24； d/D = 0.46, RMSE = 0.235;
# d/D = 0.518, RMSE = 0.222; d/D = 0.56, RMSE = 0.226; d/D = 0.6, RMSE = 0.22;

import matplotlib.pyplot as plt
rp_sketching_d = [0.1, 0.15, 0.25, 0.3, 0.4, 0.5, 0.52]
rp_sketching_RMSE = [0.86, 0.6, 0.42, 0.36, 0.3, 0.267, 0.252]

ac_sketching_d = [0.15, 0.2, 0.235, 0.285,  0.4, 0.45, 0.5, 0.52]
ac_sketching_RMSE = [0.78, 0.61, 0.39, 0.35,  0.278, 0.275, 0.27, 0.2]

us_sketching_d = [0.087, 0.137, 0.195, 0.23, 0.3, 0.518, 0.56, 0.6]
us_sketching_RMSE = [0.58, 0.4, 0.319, 0.3, 0.27, 0.222, 0.226, 0.22]


plt.plot(rp_sketching_d, rp_sketching_RMSE,  marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4, label = 'JL_sketching')
plt.plot(ac_sketching_d, ac_sketching_RMSE, marker='x', color='red', linewidth=2, label = 'Censoring_based_sketching')
plt.plot(us_sketching_d, us_sketching_RMSE, marker='*', color='orange', linewidth=2, label = 'Censoring_based_sketching')
plt.legend()
plt.savefig('img1.png')
plt.show()