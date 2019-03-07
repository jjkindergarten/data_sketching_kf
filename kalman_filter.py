#%% prediction step

from numpy import dot
def kf_predict(theta, P, F, Q):
    theta = dot(F, theta)
    P = dot(F, dot(P, F.T)) + Q
    return(theta,P)

#%% correlation step

from numpy import dot, sum, tile, linalg, identity
from numpy.linalg import inv

def kf_update(Theta, P, Y, X, R):
    IM = dot(X, Theta)
    IS = R + dot(X, dot(P, X.T))
    K = dot(P, dot(X.T, inv(IS)))
    Theta = Theta + dot(K, (Y-IM))
    P = dot((identity(len(P)) - dot(K, X)), P)

    return (Theta,P)



