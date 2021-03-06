import numpy as np
from scipy.linalg import cho_factor, cho_solve
from ship import Ship


def lm(f, x_data, y_data, par, sigma=None, verbose=False, jac=None):
    max_it = 100
    lam = 1e-2
    down_factor = 0.5
    up_factor = 3
    ftol = 1e-10  # Допуск для дельта err
    i = 0  # Число итераций
    nf = 1  # Число вычислений функции

    f_par = f(x_data, par)
    err = err_func(x_data, y_data, par, f_par, sigma)

    while i < max_it:
        if jac is not None:
            J = jac(x_data, par)
        else:
            J = numeric_jac(f, x_data, par, f_par)
            nf += 4

        if sigma is None:
            b = J.T.dot(y_data - f_par)
        else:
            for i in range(len(par)):
                J[:, i] /= sigma
            b = J.T.dot((y_data - f_par) / sigma)

        H = J.T.dot(J)

        step = False

        while (not step) and (i < max_it):
            try:
                A = H + lam * np.diag(np.diag(H))  # Fletcher modification
                # A = H + lam * np.eye(len(par)) # Standart LM

                L, low = cho_factor(A)
                delta_par = cho_solve((L, low), b)

                new_par = par + delta_par
                f_par = f(x_data, new_par)
                nf += 1
                new_err = err_func(x_data, y_data, new_par, f_par, sigma)
                delta_err = err - new_err
                step = delta_err >= 0.

                if not step:
                    lam *= up_factor

            except(np.linalg.LinAlgError):
                lam *= up_factor

        par = new_par
        err = new_err
        i += 1

        if (verbose):
            print('it = {},   lambda = {:.2e}, err = {:.4f}, par = {}, se = {}'.format(
                i, lam, err, _format_par(par), np.degrees(np.sqrt(err / len(y_data)))))

        lam *= down_factor

        if delta_err < ftol:
            break

    if jac is not None:
        J = jac(x_data, par)
    else:
        J = numeric_jac(f, x_data, par, f_par)

    H = J.T.dot(J)

    return par, np.linalg.inv(H) * err / (len(y_data) - len(par)), [nf, i]


def err_func(x_data, y_data, par, f_par, sigma):
    res = f_par - y_data
    if sigma is not None:
        err = np.sum(res ** 2 / sigma)
    else:
        err = np.sum(res ** 2)
    return err


def numeric_jac(f, x_data, par, f_par):
    h = 1e-5
    n = len(par)
    J = []
    for i in range(n):
        d = np.zeros(n)
        d[i] = h
        grad = (f(x_data, par + d) - f_par) / h
        J.append(grad)
    return np.array(J).T


def _format_par(par):
    res = []
    res.append(np.degrees(Ship.transform_to_bearing(par[0])))
    res.append(par[1])
    res.append(np.degrees(Ship.transform_to_bearing(par[2])))
    res.append(par[3])
    return res
