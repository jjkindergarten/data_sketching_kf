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

def noise_cov(dim):
    import numpy as np
    cov = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            cov[i,j] = 0.5**np.abs(i-j)
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


if __name__ == "__main__":
    p = 50
    N = 100
    D = 500
    sigma_state_noise = 0.01
    sigma_sample_noise = 1



