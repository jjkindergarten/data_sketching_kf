# %%
# estimate performance
def per_RSME(es_theta, theta):
    from numpy.linalg import norm
    return norm(es_theta - theta)

def relative_error(es_theta, theta, X, y):
    from numpy import dot
    from numpy.linalg import norm
    error_preditct = y - dot(X, theta)
    error_noise = y - dot(X, es_theta)
    return norm(error_preditct)/norm(error_noise)
