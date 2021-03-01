import numpy as np
from scipy.linalg import cho_factor, cho_solve


def lm(f, x_data, y_data, par, sigma = None, verbose = False):
    max_it = 1
    lam = 1e-2
    down_factor = 0.5
    up_factor = 3
    ftol = 1e-8
    i = 1

    f_par = f(x_data, par)
    err = err_func(x_data, y_data, par, f_par, sigma)

    while i < max_it:
        J = jacobian_func(f, x_data, y_data, par, f_par)

        if sigma is None:
            b = J.transpose().dot(y_data - f_par)
        else:
            for i in range(len(par)):
                J[:, i] /= sigma
            b = J.transpose().dot((y_data - f_par) / sigma)

        H = J.transpose().dot(J)

        step = False

        while (not step) and (i < max_it):
            try:
                A = H + lam * np.diag(np.diag(H)) # Fletcher modification
                # A = H + lam * np.eye(len(par))

                L, low = cho_factor(A)
                delta_par = cho_solve((L, low), b)

                # delta_par = np.linalg.solve(A, b)

                new_par = par + delta_par
                f_par = f(x_data, new_par)
                new_err = err_func(x_data, y_data, new_par, f_par, sigma)
                delta_err = err - new_err
                step = delta_err >= 0.

                if not step:
                    lam *= up_factor
                    i += 1

            except(np.linalg.LinAlgError):
                print('blin')
                lam *= up_factor

        par = new_par
        err = new_err
        lam *= down_factor
        i += 1

        if (verbose):
                    print('it = {},   lambda = {:.2e}, err = {:.4f}, par = {}'.format(
                        i, lam, err, par))

        if delta_err < ftol:
            break

    J = jacobian_func(f, x_data, y_data, par, f_par)
    H = J.transpose().dot(J)

    return par, np.linalg.inv(H) * err / (len(y_data) - len(par)), i


def err_func(x_data, y_data, par, f_par, sigma):
    err = 0.
    res = f_par - y_data
    if sigma is not None:
        err = np.sum(res ** 2 / sigma)
    else:
        err = np.sum(res ** 2)
    return err


def jacobian_func(f, x_data, y_data, par, f_par):
    h = 1e-5
    n = len(par)
    J = []
    for i in range(n):
        d = np.zeros(n)
        d[i] = h
        grad = (f(x_data, par + d) - f_par) / h
        J.append(grad)
    return np.array(J).transpose()
