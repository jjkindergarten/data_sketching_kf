def EKF_LWR(rho_initial, steps, dt, route, vff, rhoJ, rhoC):
    import numpy as np
    dim = len(rho_initial)
    percent_dav_state = 0.1
    P = percent_dav_state**2 * np.identity(dim)
    P[0,0] = 0
    P[-1, -1] = 0 # initial cov matrix
    percent_dev_means = 0.05 # percentage of dev of the measurement

    J, w, d = computePolyParam(dt, route, vff, rhoJ, rhoC)

    rho = np.zeros([dim, steps + 1])
    currentRho = rho_initial
    rho[:1] = currentRho
    measSteps = 0

    for j in range(steps):
        print("time step: {} - {}".format(j+1, steps))
        currentRho, P = KPapriorMode(currentRho, P, s2m(rho2s(currentRho, d)),
                                     percent_dav_state, J, w, rhoJ)

    if ((j-1) % 6) == 0:
        measSteps = measSteps + 1
        Hj = route.observationMatrix[route.activeSensors[measSteps],:]
        Hj = [np.zeros(Hj.shape[0], 1), Hj, np.zeros(Hj.shape[0], 1)]
        measurements = route.densityMeasured[:, measSteps]
        rhoMeasured = Hj @ np.array(0, measurements, 0).T
        currentRho, P = KFaposteriori(rho, P, rhoMeasured, Hj, rhoJ, percent_dev_means)

    rho[:, j+1] = currentRho





def computePolyParam(dt, route, vff, rhoJ, rhoC):
    import numpy as np

    # don't know what route means
    alpha = dt / np.mean(route.cellLength)
    omega_f = vff * rhoC / (rhoJ - rhoC)

    J = np.array(
        [0, 1-alpha*omega_f, alpha*omega_f],
        [0, 1-alpha*omega_f, 0],
        [0, 1, alpha*omega_f],
        [0, 1-alpha*vff, 0],
        [alpha*vff, 1, alpha*omega_f],
        [alpha*vff, 1, 0],
        [alpha*vff, 1-alpha*vff, 0]
    )

    w = np.array(
        [0], 
        [alpha*omega_f*rhoC], 
        [-alpha*omega_f*rhoC], 
        [alpha*vff*rhoC],
        [-alpha*omega_f*rhoJ],
        [-alpha*vff*rhoC],
        [0]
    )

    d = np.array(
        [(rhoJ - rhoC) / rhoC, 1, -rhoJ],
        [1, 0, -rhoC],
        [0, 1, -rhoC]
    )

    return J, w, d

def KPapriorMode(rho, P, m, percentStateNoise, J, w, rhoJ):
    import numpy as np

    dim = len(rho)

    # update rho
    rhoNext = np.array([rho[1]], np.zeors([dim-2, 1]), [rho(dim)])
    for i in range(1, dim-1):
        rhoNext[i] = J[m[i-2], :] @ np.array([rho[i-1]], [rho[i]], [rho[i+1]]) + w[m[i-2]]
    
    rhoNext = max(min(rhoNext, rhoJ), 0)

    # update P
    Pnext = np.zeros([dim, dim])
    temp = Pnext
    for i in range(1, dim-1):
        for j in range([2, dim-1]):
            temp[i,j] = J[m[i-2], :] @ np.array([P(i-1,j)], [P(i,j)], [P(i,j+1)])

    for i in range(2, dim-1):
        for j in range(2, dim-1):
            Pnext[i, j] = np.array([temp[i, j-1]], [temp[i,j]], [temp[i, j+1]]) @ J[m[j-2], :]
    Pnext = Pnext + percentStateNoise**2*np.diag(rho**2)
    return rhoNext, Pnext

def rho2s(rho, d):
    import numpy as np
    dim = len(rho)
    s = np.zeros([dim-1, 1])
    for i in range(dim-1):
        ind = (d @ np.array([rho[i]], [rho[i+1]], [1])) > 0
        temp = np.array(ind[1]*ind[3], ind[2]*(1-ind[3]), (1-ind[1])*(1-ind[2]))
        s[i] = temp[temp > 0]

        return s

def s2m(m):
    import numpy as np
    dim = len(m) + 2
    s = np.zeros([dim-1, 1])
    temp = np.array(
        (m[1] == 1 | m[2] == 2),
        (m[1] == 3 | m[1] == 4),
        (m[1] == 5 | m[1] == 6 | m[1] == 7)
    )
    s[1] = temp[temp > 0]

    for i in range(dim-2):
        temp = np.array(
            (m[i] == 1 | m[i] == 3 | m[i] == 5),
            (m[1] == 2 | m[i] == 6),
            (m[1] == 4 | m[i] == 7)
        )
        s[i+1] = temp[temp > 0]
    return s

def KFaposteriori(rho, P, rhoMeasured, H, rhoJam, percentErrorMeasured):

    import numpy as np
    R = percentErrorMeasured**2 * np.diag(rhoMeasured**2)
    residualCov = H @ P @ H.T + R + 0.001 * np.identity(R.shape[0])
    gain = P @ H.T @ np.linalg.inv(residualCov)

    residual = rhoMeasured - H*rho
    rhoNext = rho + gain @ residual
    rhoNext = np.max(np.min(rhoJam, rhoNext), 0)

    Pnext = (np.identity(len(rho) - gain @ H)) @ P

    return rhoNext, Pnext






