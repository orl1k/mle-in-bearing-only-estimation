import numpy as np
import pandas as pd
import tma.helper_functions as f
from tma.lm import lev_mar
from functools import wraps, partial
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
    def get_df(result):
        mapper = {
            "b0": "П0_ист",
            "d0": "Д0_ист",
            "c0": "К0_ист",
            "v0": "V0_ист",
            "res_b0": "П0_расч",
            "res_d0": "Д0_расч",
            "res_c0": "К0_расч",
            "res_v0": "V0_расч",
            "init_b0": "П0_апр",
            "init_d0": "Д0_апр",
            "init_c0": "К0_апр",
            "init_v0": "V0_апр",
            "cur_b0": "Птек_ист",
            "cur_d0": "Дтек_ист",
            "cur_res_b0": "Птек_расч",
            "cur_res_d0": "Дтек_расч",
            "std_x": "СКО X",
            "std_y": "СКО Y",
            "std_vx": "СКО VX",
            "std_vy": "СКО VY",
            "ka": "Ка",
            "kb": "Кб",
            "kc": "Успех",
            "t": "Время",
            "nf": "Вычисления",
            "iter": "Итерации",
        }

        if isinstance(result, list):
            result_list = map(Algorithms.parser, result)
        else:
            result_list = map(Algorithms.parser, [result])

        return pd.DataFrame(result_list).rename(columns=mapper)

    @staticmethod
    def parser(result):
        parsed_res = namedtuple(
            "res",
            [
                "b0",
                "d0",
                "c0",
                "v0",
                "res_b0",
                "res_d0",
                "res_c0",
                "res_v0",
                "init_b0",
                "init_d0",
                "init_c0",
                "init_v0",
                "cur_b0",
                "cur_d0",
                "cur_res_b0",
                "cur_res_d0",
                "std_x",
                "std_y",
                "std_vx",
                "std_vy",
                "ka",
                "kb",
                "kc",
                "t",
                "nf",
                "iter",
            ],
        )

        return parsed_res(*(i for sublist in result for i in sublist))

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


class Swarm(Algorithms):
    def __init__(self, model, seeded=True):
        super().__init__(model)
        self._model = model
        self._seeded = seeded
        self._target = None
        self._alg_dict = {
            "ММП2": super().mle_v1,
            "ММП": super().mle_v2,
            "N пеленгов": super().n_bearings,
            "ДММП": super().dynamic_mle,
        }

    def set_target(self, target):
        self._target = target

    def set_target_func(self, target_func):
        self._target_func = target_func

    def set_algorithm(self, algorithm="ММП"):
        self._algorithm = self._alg_dict[algorithm]

    def set_initial(self, initial=None):
        if initial is None:
            self._get_initial = lambda x: [0.0, 25.0, 90.0, 7.0]
        else:
            self._get_initial = lambda x: initial

    def set_initial_func(self, initial_func):
        self._get_initial = (
            initial_func if self._seeded else lambda x: initial_func(seed=x)
        )

    def set_noise_func(self):
        self._update_noise = lambda x: self.model.set_noise(seed=x)

    def create_swarm(self):
        self.set_noise_func()
        if self._target is None:
            self._update_target = lambda x: self._model.new_target(
                p0=self._target_func(seed=x)
            )
        else:
            self._update_target = lambda x: None

    def run(self, n=100):
        self.create_swarm()
        res_arr = []

        algorithm = self._algorithm
        iterator = range(n) if self._seeded else (None for _ in range(n))
        initial_func = self._get_initial
        update_target = self._update_target
        update_noise = self._update_noise
        convert_initial = f.convert_to_xy

        for i in iterator:
            update_target(i)
            update_noise(i)
            initial = convert_initial(initial_func(i))
            result = algorithm(initial)
            res_arr.append(result)

        return res_arr