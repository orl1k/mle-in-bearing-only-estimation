import numpy as np
from tma.helper_functions import convert_to_bdcv
from scipy.linalg import cho_factor, cho_solve


def lev_mar(
    f,
    x_data,
    y_data,
    par,
    sigma=None,
    verbose=False,
    jac=None,
    lam=1e-2,
    down_factor=0.5,
    up_factor=3,
    max_it=1000,
    ftol=1e-8,
):
    i = 0  # Число итераций
    nf = 1  # Число вычислений функции
    status = -1
    if jac is None:
        jac_ = numeric_jac
    else:
        jac_ = lambda f, x_data, par, f_par: jac(x_data, par)

    if verbose:
        statuses = {
            0: [
                f"Разность ошибок суммы квадратов "
                + "невязок меньше ftol = {ftol:.0e}"
            ],
            1: ["Число итераций превысило свой лимит max_it = %i" % max_it],
            -1: ["Необработанный выход"],
        }
        print(f"lam: {lam}, lam_up: {up_factor}, lam_down: {down_factor}")

    f_par = f(x_data, par)
    err = err_func(x_data, y_data, par, f_par, sigma)

    while i < max_it:
        J = jac_(f, x_data, par, f_par)

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
                A = H + lam * np.diag(np.diag(H))  # Marquardt modification
                # A = H + lam * np.eye(len(par)) # Standart LM

                L, low = cho_factor(A)
                delta_par = cho_solve((L, low), b)

                new_par = par + delta_par
                f_par = f(x_data, new_par)
                nf += 1
                new_err = err_func(x_data, y_data, new_par, f_par, sigma)
                delta_err = err - new_err
                step = delta_err >= 0.0

                if verbose:
                    verbose_func(
                        i, lam, delta_err, delta_par, par, b, err, y_data
                    )

                if not step:
                    lam *= up_factor

            except np.linalg.LinAlgError:
                lam *= up_factor

        par = new_par
        err = new_err
        i += 1

        lam *= down_factor

        if delta_err < ftol:
            status = 0
            break

    if status == -1:
        status = 1

    if verbose:
        print(statuses[status][0])

    J = jac_(f, x_data, par, f_par)
    H = J.T.dot(J)

    try:
        return par, np.linalg.inv(H) * err / (len(y_data) - len(par)), [nf, i]
    except np.linalg.LinAlgError:
        return par, np.nan * np.ones(shape=(4, 4)), [nf, i]


def err_func(x_data, y_data, par, f_par, sigma):
    res = f_par - y_data
    # res = (res + np.pi) % (2 * np.pi) - np.pi # normalization
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


def verbose_func(i, lam, delta_err, delta_par, par, b, err, y_data):
    print(
        f"it: {i} lam: {lam:.2e} "
        + f"err: {err:.4f} par: {convert_to_bdcv(par)} "
        + f"std: {np.degrees(np.sqrt(err / len(y_data))):.2f}"
    )
