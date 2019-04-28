

import numpy as np
from scipy.linalg import orth, qr, inv, solve_triangular

def spd_sys_approximate(A, k, l, Orthogonalization = False):
    # r < k < l and k < n and l < m

    m,n = A.shape
    Omega = np.random.randn(n, k)
    Psi = np.random.randn(l, m)

    if Orthogonalization:
        Omega = orth(Omega)
        Psi = orth(Psi)

    Y = A @ Omega
    W = Psi @ A

    Q, _ = qr(Y, mode='economic')

    Psi_Q = Psi @ Q
    U, T = qr(Psi_Q, mode='economic')



    X = inv(T) @ (U.T @ W)

    # X = solve_triangular(T, np.identity(k88

    U_sys, T_sys = qr(np.concatenate((Q, X.T), axis=1), mode='economic')
    T1 = T_sys[:, :k]
    T2 = T_sys[:, (k):]
    S = (T1 @ T2.T + T2 @ T1.T)/2

    D,V = np.linalg.eig(S)
    U_spd = U_sys @ V

    D[D <0]=0
    D = np.diag(D)
    A = U_spd @ D @ U_spd.T

    return A

def create_low_rank_matrix(size = (100, 100), rank = 10, symmetry = True):
    import numpy as np
    A = np.random.randn(size[0], size[1])
    u, s, d = np.linalg.svd(A)
    s[rank:] = 0
    s_diag = np.diag(s)
    A_low_rank = u @ s_diag @ d

    if symmetry:
        A_low_rank = (A_low_rank + A_low_rank.T)/2

    A_low_rank =  A_low_rank @ A_low_rank

    return A_low_rank











