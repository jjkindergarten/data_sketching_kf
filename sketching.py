#%%
# RP sketcching module

c = 1 / 2 ** 0.5
def fht(x,n):          # T(n) flops
    import numpy as np
    if n == 1:         # 0 flops
        return x       # 0 flops
    m = int(n/2)            # 0 flops
    ht = fht(x[:m],m)  # T(n/2) flops
    hb = fht(x[m:],m)  # T(n/2) flops
    y = np.concatenate([ht, ht])      # 0 flops
    y[:m] = y[:m] + hb # n/2 flops
    y[m:] = y[m:] - hb # n/2 flops
    return c*y         # n flops

def generate_s_d(D, d):
    import numpy as np
    S = np.zeros((d,D))
    selection = np.random.permutation(range(D))[:d]
    S[range(d), selection] = 1
    return S

def generate_Lambda(D):
    import numpy as np
    from numpy.random import choice
    diag_value = choice([1,-1], D)*1/D**0.5
    return np.diag(diag_value)


def fjl_sketching(X, y, R, d):
    from synthetic_data_generation import random_sample
    import numpy as np
    from scipy.linalg import hadamard
    D = X.shape[0]

    lmd = generate_Lambda(D)
    h = hadamard(D)
    S = generate_s_d(D, d)

    sketch = np.dot(S, np.dot(h, lmd))

    X_hat = np.dot(sketch, X)
    y_hat = np.dot(sketch, y)
    R_hat = np.dot(sketch, np.dot(R, sketch.T))

    return X_hat, y_hat, R_hat

def build_s(row_k, ncol):
    import numpy as np
    from numpy.random import randint
    from scipy.sparse import coo_matrix
    h_i = randint(0, row_k, ncol)
    t_radn = randint(0, 1+1, ncol) * 2-1
    # S = coo_matrix((t_radn, (h_i, np.array(range(ncol)))), shape = (row_k, ncol))
    S = np.zeros((row_k, ncol))
    for i in range(ncol):
        S[h_i[i], i] = t_radn[i]
    return S

#%% AC sketching

def update_theta(theta, x_i, y_i, mu):
    import numpy as np
    theta = theta + mu * (y_i- np.dot(x_i.T, theta))
    return theta.reshape(theta.shape[0], 1)





def AC_sketching(theta, y, X, R, tau, mu):
    import numpy as np
    S = []
    for i in range(y.shape[0]):
        c = 1 * (y[i,0] - np.dot(X[i,:], theta) <= tau/(R[i,i]**0.5))
        if c == 0:
            S.append(i)
            x_i = X[i]
            y_i = y[i,0]
            theta = update_theta(theta, x_i, y_i, mu)
    R_S = R[S,:]
    R_S = R_S[:,S]
    return y[S], X[S], R_S, len(S)

def error(y_i, x_i, theta):
    import numpy as np
    return y_i - np.dot(x_i, theta)

def generate_kn(P, x, s):
    import numpy as np
    return np.dot(P, x) * 1/s

def generate_s(x_i, P, sigma):
    import numpy as np
    return np.dot(x_i, np.dot(P, x_i)) + sigma**2

def update_predict_p(P, x_i, s):
    import numpy as np
    px = np.dot(P, x_i)
    return P - np.dot(px, px.T)/s

def generate_mu(x_i, gamma, sigma):
    from numpy.linalg import norm
    mu = gamma + sigma**2
    mu = norm(x_i)**2 * mu
    return gamma/mu









