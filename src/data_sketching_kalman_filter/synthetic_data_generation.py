# state transition matrix

def state_transition_matrix(p):
    import numpy as np
    F = np.zeros((p,p))
    F[p-1,0] = 1
    for i in range(p-1):
       F[i,i+1] = 1
    return F


def noise_generation(p, cov):
    import numpy as np
    normal_var = cov
    w = np.random.multivariate_normal(mean= np.zeros((p)), cov = normal_var)
    return w.reshape((p,1))

def noise_cov(dim, detect_2power = False):
    import numpy as np
    from math import ceil
    if detect_2power:
        dim_2power = ceil(np.log2(dim))
        cov = np.zeros((2**dim_2power, 2**dim_2power))
    else:
        cov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cov[i,j] = 0.5**np.abs(i-j)
    return cov

def noise_cov_idp(dim, detect_2power = False):
    import numpy as np
    from math import ceil
    if detect_2power:
        dim_2power = ceil(np.log2(dim))
        cov = np.zeros((2**dim_2power, 2**dim_2power))
    else:
        cov = np.zeros((dim, dim))
    for i in range(dim):
        cov[i,i] = 1
    return cov


def initial_state(p):
    import numpy as np
    p0 = np.identity(p) * 0.04
    m0 = np.zeros(p)
    m0[0] = 20
    m0[4] = -30
    theta = np.random.multivariate_normal(mean= m0, cov = p0)
    return theta.reshape((p,1))

def random_sample(p, D):
    import numpy as np

    matrix_x = np.zeros((D,p))
    for i in range(D):
        alpha = np.random.choice([0.5, 1.5], 1)
        matrix_x[i,:] = alpha*np.random.multivariate_normal(mean=np.zeros(p), cov = np.identity(p))
    return matrix_x

def synthetic_data_generate(N, p, Q, R, F, D, theta, detect_2power = False):
    import numpy as np
    from math import ceil
    w_set = []
    v_set = []
    theta_set = []
    X_set = []
    y_set = []

    if detect_2power:
        D_2power = ceil(np.log2(D))
        D_rest = 2**D_2power -D
    for i in range(N):
        w = noise_generation(p,  Q)
        v = noise_generation(D,  R)
        theta = np.dot(F, theta) + w
        X = random_sample(p, D)
        y = np.dot(X, theta) + v

        if detect_2power:
            X = np.vstack((X, np.zeros([D_rest, p])))
            y = np.vstack((y, np.zeros([D_rest, 1])))

        w_set.append(w)
        v_set.append(v)
        theta_set.append(theta)
        X_set.append(X)
        y_set.append(y)

    return w_set,v_set, theta_set, X_set, y_set




if __name__ == "__main__":
    p = 50
    N = 100
    D = 500
    sigma_state_noise = 0.01
    sigma_sample_noise = 1



