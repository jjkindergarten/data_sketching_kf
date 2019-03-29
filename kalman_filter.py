#%% prediction step

from numpy import dot
def kf_predict(theta, P, F, Q):
    theta = dot(F, theta)
    P = dot(F, P).dot(F.T) + Q
    return(theta,P)

#%% correlation step

from numpy import dot, sum, tile, linalg, identity
from numpy.linalg import inv

def kf_update(Theta, P, Y, X, R):
    IM = dot(X, Theta)
    IS = dot(X, P).dot(X.T) + R
    K = dot(P, X.T).dot(inv(IS))
    Theta = Theta + dot(K, (Y-IM))
    P = (identity(len(P)) - dot(K, X)).dot(P)

    return (Theta,P)



