from __future__ import division, print_function, absolute_import

import numpy as np

from scipy._lib._util import check_random_state
from scipy.sparse import csc_matrix
from numpy.linalg import lstsq

def cwt_matrix(n_rows, n_columns, seed=None):

    rng = check_random_state(seed)
    rows = rng.randint(0, n_rows, n_columns)
    cols = np.arange(n_columns+1)
    signs = rng.choice([1, -1], n_columns)
    S = csc_matrix((signs, rows, cols),shape=(n_rows, n_columns))
    return S

def clarkson_woodruff_transform(input_matrix, sketch_size, seed=None):

    S = cwt_matrix(sketch_size, input_matrix.shape[0], seed)
    return S.dot(input_matrix)

def sketching_projection(x, A, b, s_size):
    S = cwt_matrix(s_size, A.shape[0])
    A_s = S.dot(A)
    b_s = S.dot(b)
    r = np.dot(A_s,x) - b_s
    return x - lstsq(A_s, r, rcond=None)[0]

if __name__ == '__main__':
    A = np.random.rand(100, 50)
    x = np.random.rand(50,1)
    b = A @ x

    x_test = np.zeros((50, 1))
    for t in range(100):
        x_test = sketching_projection(x_test, A, b, 0.4)
        print(np.linalg.norm(b - A@x_test, 'fro'))

