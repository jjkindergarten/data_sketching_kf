import numpy as np

# parameter set up
tau = 0.01
N = 100

def in_Theta(theta_predict_ahead, theta_predict, F, Q):
    import numpy as np
    theta_dis = (theta_predict_ahead - np.dot(F, theta_predict))
    region = np.dot(theta_dis.T, np.dot(Q, theta_dis))
    return region

def generate_B(P_predict, P_predict_ahead, F):
    import numpy as np
    from numpy.linalg import inv
    B = np.dot(P_predict, np.dot(F, inv(P_predict_ahead)))
    return B


smooth_theta_predict = []
smooth_P_predict = []
for n in range(start=N-1, stop=0):
    region = in_Theta(theta_predict[n+1], theta_predict[n], F, Q, tau)
    if region <= tau:
        smooth_theta_predict.append(theta_predict[n])
        smooth_P_predict.append(P_predict)
    else:
        B = generate_B(P_predict, P_predict_ahead, F)

        smooth_theta = theta_predict[n] + np.dot(B, (theta_predict[n+1] -  np.dot(F, theta_predict[n])))
        smooth_theta_predict.append(smooth_theta)

        P_error = smooth_P_predict[n] - P_predict[n]
        smooth_P_predict = P_predit[n] + np.dot(B, np.dot(P_error, B.T))




