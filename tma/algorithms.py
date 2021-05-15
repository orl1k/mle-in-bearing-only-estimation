import numpy as np
import tma.helper_functions as f
from tma.lm import lev_mar
from functools import wraps
from time import perf_counter
from collections import namedtuple
from scipy.optimize import curve_fit


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = perf_counter()
        result = f(*args, **kwargs)
        elapsed = perf_counter() - start_time
        return elapsed, *result

    return wrapper


def full_output(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        result = f(self, *args, **kwargs)
        return self._get_result(f.__name__, *result)

    return wrapper


class Algorithms:
    def __init__(self, model):
        self.model = model
        self.D = (
            (1 / 0.5, 1 / 0.05, 1 / 5, 1 / 0.05),
            (1 / 1, 1 / 0.1, 1 / 10, 1 / 0.1),
            (1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1),
            (1 / 1, 1 / 0.15, 1 / 10, 1 / 0.15),
            (1 / 1, 1 / 0.2, 1 / 15, 1 / 0.2),
        )
        self.alg_dict = {
            "ММП2": self.mle_v1,
            "ММП": self.mle_v2,
            "N пеленгов": self.n_bearings,
            "ДММП": self.dynamic_mle,
        }

    @full_output
    @timing
    def n_bearings(self, *args):

        b0 = self.model.bearings_with_noise[0]
        n = len(self.model.observer_data[0])

        bearings = self.model.bearings_with_noise
        bearing_origin = np.array([b0] * n)

        H = [-np.sin(bearing_origin - bearings) * 1000.0]
        H.append(np.sin(bearings) * self.model.observer_data[2])
        H.append(-np.cos(bearings) * self.model.observer_data[2])
        H = np.array(H).T

        d = self.model.observer_data[0] * np.sin(
            bearings
        ) - self.model.observer_data[1] * np.cos(bearings)

        res = np.linalg.solve(H.T.dot(H), H.T.dot(d))
        res = np.insert(res, 0, b0)

        b, d = b0, res[1]
        res[0:2] = f.convert_from_polar(d, b)

        return [res]

    @full_output
    @timing
    def mle_v1(self, p0):

        fun = lambda data, b, d, c, v: f.xy_func(data, [b, d, c, v])

        res, cov, score = curve_fit(
            fun,
            self.model.observer_data,
            self.model.bearings_with_noise,
            p0=p0,
            full_output=True,
        )[0:3]

        score = score["nfev"]

        return res, cov, [score, np.nan], p0

    @full_output
    @timing
    def mle_v2(self, p0, verbose=False):

        res, cov, nfi = lev_mar(
            f.xy_func,
            self.model.observer_data,
            self.model.bearings_with_noise,
            p0,
            verbose=verbose,
            jac=f.xy_func_jac,
        )

        return res, cov, nfi, p0

    def dynamic_mle(self, p0):
        t = [420, 660, 1200]
        res_arr = []
        lam = 1e-2
        from copy import deepcopy

        for i in t:
            self.model_t = deepcopy(model)
            self.model_t.end_t = i
            self.model_t.bearings = self.model.bearings[
                : (i // self.model.tau + 1)
            ]
            self.model_t.target_data = self.model.target_data[
                :, : (i // self.model.tau + 1)
            ]
            self.model_t.observer_data = self.model.observer_data[
                :, : (i // self.model.tau + 1)
            ]
            self.model_t.bearings_with_noise = self.model.bearings_with_noise[
                : (i // self.model.tau + 1)
            ]
            self.model_t.distances = self.model.distances[
                : (i // self.model.tau + 1)
            ]
            res = lev_mar(
                f.xy_func,
                self.model_t.observer_data,
                self.model_t.bearings_with_noise,
                p0,
                jac=f.xy_func_jac,
            )
            res_arr.append(
                _get_result(
                    self.model_t,
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
        self,
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

        if fixed_target:
            target_f = lambda x: None
        else:
            target_func = (
                target_func
                if target_func is not None
                else self.model.get_random_p0
            )
            target_f = lambda x: self.model.new_target(p0=target_func(seed=x))

        if p0 is not None:
            p0_f = lambda x: p0
        else:
            if seeded:
                p0_f = lambda x: self.model.get_random_p0(seed=x + 100000)
            else:
                p0_f = self.model.get_random_p0

        noise_f = lambda x: self.model.set_noise(seed=x)

        algorithm = self.alg_dict[algorithm_name]
        iterator = range(n) if seeded else [None] * n

        for i in iterator:
            target_f(i)
            noise_f(i)
            result = algorithm(f.convert_to_xy(p0_f(i)))
            res_arr.append(result)

        return res_arr

    def _get_result(self, algorithm_name, t, res, *args):

        try:
            cov, nfi, p0 = args
            perr = self.handle_cov_perr(cov)
            p0_res = f.convert_to_bdcv(p0)
        except ValueError:
            perr = p0_res = [np.nan] * 4
            nfi = [np.nan] * 2

        r = self.model.bearings_with_noise - f.xy_func(
            self.model.observer_data, res
        )

        b_end_pred = np.degrees(
            f.to_bearing(f.xy_func(self.model.observer_data[:, -1], res))
        )

        r_x_end = (
            self.model.observer_data[0][-1]
            - 1000.0 * res[0]
            - res[2] * self.model.end_t
        )

        r_y_end = (
            self.model.observer_data[1][-1]
            - 1000.0 * res[1]
            - res[3] * self.model.end_t
        )

        d_end_pred = np.sqrt(r_x_end ** 2 + r_y_end ** 2) / 1000.0

        res = f.convert_to_bdcv(res)

        self.model.last_result = res
        r = np.degrees(r)
        k_a = np.sum(r ** 2) / len(r)

        true_res = f.convert_to_bdcv(self.model.true_params)
        d = (1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1)
        delta = self._get_delta(d, b_end_pred, d_end_pred, res, true_res)

        k_b = np.sum(delta) / 4
        k_c = []

        for d in self.D:
            delta = self._get_delta(d, b_end_pred, d_end_pred, res, true_res)
            k_c.append(int(all(delta < [1, 1, 1, 1])))

        current_values = [
            np.degrees(f.to_bearing(self.model.bearings[-1])),
            self.model.distances[-1],
            b_end_pred,
            d_end_pred,
        ]

        result = namedtuple(
            algorithm_name,
            [
                "true_params",
                "result",
                "initial_value",
                "current_values",
                "params_std",
                "ka_kb_kc",
                "time",
                "nf_iter",
            ],
        )

        return result(
            true_res,
            res,
            p0_res,
            current_values,
            perr,
            [k_a, k_b, k_c],
            [t],
            nfi,
        )

    def _get_delta(self, d, b_end_pred, d_end_pred, res, true_res):
        b_end = np.degrees(f.to_bearing(self.model.bearings[-1]))
        d_end = self.model.distances[-1]
        delta = list(d)
        delta[1] /= d_end
        delta[3] /= true_res[3]
        temp = true_res - res
        temp[0:2] = abs(b_end_pred - b_end), abs(d_end_pred - d_end)
        temp[0] = (temp[0] + 180) % 360 - 180
        temp[2] = (temp[2] + 180) % 360 - 180
        return abs(temp) * delta

    @staticmethod
    def handle_cov_perr(cov):
        diag = np.diag(cov)
        if (diag < 0).any():
            perr = np.empty(4)
            perr[:] = np.nan
        else:
            perr = np.sqrt(diag)
        return perr

    @staticmethod
    def print_result(result, presicion=4):
        d = result._asdict()
        print("-" * 79)
        print(f"results for {result.__class__.__name__} algorithm:")
        d["current_values"] = np.array(d["current_values"])
        d["ka_kb_kc"][0:2] = [f"{i:.2f}" for i in d["ka_kb_kc"][0:2]]
        d["time"] = f"{d['time'][0]:.4f}"
        with np.printoptions(precision=presicion, suppress=True):
            for key, value in d.items():
                print(key + ":", value)
        print("-" * 79)
