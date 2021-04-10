import numpy as np
import tma.functions as f
from tma.lm import lev_mar
from scipy.optimize import curve_fit
import time


def n_bearings_algorithm(model, p0):
    algorithm_name = "Метод N пеленгов"

    start_time = time.perf_counter()
    b0 = model.bearings_with_noise[0]
    n = len(model.observer_data[0])

    bearings = model.bearings_with_noise
    bearing_origin = np.array([b0] * n)

    H = [-np.sin(bearing_origin - bearings) * 1000.0]
    H.append(np.sin(bearings) * model.observer_data[2])
    H.append(-np.cos(bearings) * model.observer_data[2])
    H = np.array(H).T
    d = model.observer_data[0] * np.sin(bearings) - model.observer_data[1] * np.cos(
        bearings
    )

    res = np.linalg.solve(H.T.dot(H), H.T.dot(d))

    # reg = LinearRegression().fit(H, d)
    # res = reg.coef_

    res = np.insert(res, 0, b0)
    b, d = b0, res[1]
    res[0] = d * np.cos(b)
    res[1] = d * np.sin(b)

    stop_time = time.perf_counter()
    return get_result(
        model,
        algorithm_name,
        res.copy(),
        [np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan],
        stop_time - start_time,
    )


def mle_algorithm_v1(model, p0):
    algorithm_name = "ММП scipy"

    def fun(data, b, d, c, v):
        return f.xy_func(data, [b, d, c, v])

    start_time = time.perf_counter()
    res = curve_fit(
        fun, model.observer_data, model.bearings_with_noise, p0=p0, full_output=True
    )
    stop_time = time.perf_counter()
    score = res[2]["nfev"]
    if (np.diag(res[1]) < 0).any():
        perr = np.empty(4)
        perr[:] = np.nan
    else:
        perr = np.sqrt(np.diag(res[1]))

    return get_result(
            model,
            algorithm_name,
            res[0].copy(),
            perr,
            [score, np.nan],
            p0,
            stop_time - start_time,
        )


def mle_algorithm_v2(model, p0, verbose=False, full_output=True):
    algorithm_name = "ММП"
    start_time = time.perf_counter()
    res = lev_mar(
        f.xy_func,
        model.observer_data,
        model.bearings_with_noise,
        p0,
        verbose=verbose,
        jac=f.xy_func_jac,
        std=model.noise_std
    )
    stop_time = time.perf_counter()
    if (np.diag(res[1]) < 0).any():
        perr = np.empty(4)
        perr[:] = np.nan
    else:
        perr = np.sqrt(np.diag(res[1]))
    if full_output:
        return get_result(
            model,
            algorithm_name,
            res[0].copy(),
            perr,
            res[2],
            p0,
            stop_time - start_time,
        )
    else:
        return res[0]


def dynamic_mle(model, p0):
    algorithm_name = "ММП в реальном времени"
    t = [420, 660, 1200]
    res_arr = []
    lam = 1e-2
    from copy import deepcopy

    for i in t:
        model_t = deepcopy(model)
        model_t.end_t = i
        model_t.bearings = model.bearings[: (i // model.tau + 1)]
        model_t.target_data = model.target_data[:, : (i // model.tau + 1)]
        model_t.observer_data = model.observer_data[:, : (i // model.tau + 1)]
        model_t.bearings_with_noise = model.bearings_with_noise[: (i // model.tau + 1)]
        model_t.distances = model.distances[: (i // model.tau + 1)]
        start_time = time.perf_counter()
        res = lev_mar(
            f.xy_func,
            model_t.observer_data,
            model_t.bearings_with_noise,
            p0,
            jac=f.xy_func_jac,
            std=model.noise_std,
            return_lambda=True,
        )
        stop_time = time.perf_counter()
        res_arr.append(
            get_result(
                model_t,
                algorithm_name,
                res[0],
                np.sqrt(np.diag(res[1])),
                res[2],
                p0,
                stop_time - start_time,
            )
        )
        p0 = res[0]
        lam = res[3] * 4
    return res_arr


def swarm(
    model,
    algorithm_name="ММП",
    n=100,
    seeded=True,
    fixed_target=False,
    fixed_noise=False,
    p0_func=None,
    target_func=None,
    p0=None,
    verbose=False,
):
    res_arr = []

    if p0 is not None:
        fixed_p0 = True
        p0 = p0.copy()
    else:
        fixed_p0 = False

    if p0_func is None:
        p0_func = f.get_random_p0
    if target_func is None:
        target_func = f.get_random_p0

    alg_dict = {
        "ММП": mle_algorithm_v2,
        "Метод N пеленгов": n_bearings_algorithm,
        "ММП в реальном времени": dynamic_mle,
    }
    algorithm = alg_dict[algorithm_name]
    if fixed_p0:
        if algorithm_name in ["ММП", "ММП в реальном времени", "ММП scipy"]:
            p0[0] = f.to_angle(np.radians(p0[0]))
            p0[2] = f.to_angle(np.radians(p0[2]))
            b, d, c, v = p0
            p0 = [d * np.cos(b), d * np.sin(b), v * np.cos(c), v * np.sin(c)]
        else:
            p0[0] = f.to_angle(np.radians(p0[0]))
            p0[2] = f.to_angle(np.radians(p0[2]))
    if verbose:
        start_time = time.perf_counter()
    for i in range(n):

        if not fixed_target:
            if seeded:
                model.new_target(p0=target_func(seed=i))
            else:
                model.new_target(p0=target_func())

        if not fixed_p0:
            if seeded:
                p0 = p0_func(seed=i)
            else:
                p0 = p0_func()
            if algorithm_name in ["ММП", "ММП в реальном времени", "ММП scipy"]:
                p0[0] = f.to_angle(np.radians(p0[0]))
                p0[2] = f.to_angle(np.radians(p0[2]))
                b, d, c, v = p0
                p0 = [d * np.cos(b), d * np.sin(b), v * np.cos(c), v * np.sin(c)]
            else:
                p0[0] = f.to_angle(np.radians(p0[0]))
                p0[2] = f.to_angle(np.radians(p0[2]))

        if not fixed_noise:
            if seeded:
                model.set_noise(seed=i)
            else:
                model.set_noise()

        result = algorithm(model, p0)
        res_arr.append(result)
    if verbose:
        stop_time = time.perf_counter()
        print("Алгоритм: " + algorithm_name)
        print(
            "Моделирование {} результатов закончено за t = {:.1f} с".format(
                n, stop_time - start_time
            )
        )
    return res_arr


def get_result(model, algorithm_name, res, perr, nfev, p0, t):
    r = model.bearings_with_noise - f.xy_func(model.observer_data, res)  # residuals
    # chi_2 = sum((r) ** 2) / r.var(ddof=1)
    # r = (r + np.pi) % (2 * np.pi) - np.pi # normalization
    # if model.verbose:
    # Проверка гипотезы
    # print('z_stat = {:.2f}, p-value = {:.4f}'.format(*ztest(r, ddof=5)))
    # print('chi2_stat = {:.2f}, p-value = {:.4f}'.format(chi_2, 1 - chi2.cdf(chi_2, len(r) - 5)))
    # from matplotlib import pyplot as plt
    # plt.plot(r)
    # plt.plot(model.bearings)
    # plt.plot(f.xy_func(model.observer_data, res))
    # plt.show()

    b_end_pred = np.degrees(f.to_bearing(f.xy_func(model.observer_data[:, -1], res)))
    r_x_end = model.observer_data[0][-1] - 1000.0 * res[0] - res[2] * model.end_t
    r_y_end = model.observer_data[1][-1] - 1000.0 * res[1] - res[3] * model.end_t
    d_end_pred = np.sqrt(r_x_end ** 2 + r_y_end ** 2) / 1000.0
    res = f.convert_to_bdcv(res)

    if algorithm_name in ["ММП", "ММП в реальном времени", "ММП scipy"]:
        p0 = f.convert_to_bdcv(p0)

    model.last_result = res
    r = np.degrees(r)
    k_a = np.sum(r ** 2) / len(r)

    d = np.array([1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1])
    delta = _get_delta(model, d, b_end_pred, d_end_pred)

    k_b = np.sum(delta) / 4
    k_c = []

    D = [
        [1 / 0.5, 1 / 0.05, 1 / 5, 1 / 0.05],
        [1 / 1, 1 / 0.1, 1 / 10, 1 / 0.1],
        [1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1],
        [1 / 1, 1 / 0.15, 1 / 10, 1 / 0.15],
    ]

    for d in D:
        delta = _get_delta(model, d, b_end_pred, d_end_pred)
        k_c.append(int(all(delta < [1, 1, 1, 1])))

    result = {
        algorithm_name: {
            "Истинные параметры": model.true_params,
            "Полученные параметры": res,
            "Начальное приближение": p0,
            "Текущие значения": [
                np.degrees(f.to_bearing(model.bearings[-1])),
                model.distances[-1],
                b_end_pred,
                d_end_pred,
            ],
            "СКО параметров": perr,
            "Ка, Кб, Кс": [k_a, k_b, k_c],
            "Время работы": [t],
            "Число вычислений функции, число итераций": nfev,
        }
    }
    return result


def _get_delta(model, d, b_end_pred, d_end_pred):
    b_end = np.degrees(f.to_bearing(model.bearings[-1]))
    d_end = model.distances[-1]
    d[1] /= d_end
    d[3] /= model.true_params[3]
    temp = model.true_params - model.last_result
    temp[0] = abs(b_end_pred - b_end)
    temp[1] = abs(d_end_pred - d_end)
    temp[0] = (temp[0] + 180) % 360 - 180
    temp[2] = (temp[2] + 180) % 360 - 180
    temp = abs(temp) * d
    return temp
