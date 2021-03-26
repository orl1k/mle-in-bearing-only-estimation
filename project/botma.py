import numpy as np
from project.object import Object
import project.functions as f
from scipy.optimize import curve_fit
from project.lm import lev_mar
import time


class TMA(Object):
    def __init__(
        self,
        observer,
        target=None,
        tau=2,
        noise_mean=0,
        noise_std=np.radians(0.1),
        seed=None,
        end_t=None,
        verbose=False,
    ):
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.observer = observer
        self.observer_coords = np.array(observer.coords)
        self.tau = tau
        if end_t is None:
            end_t = len(self.observer.coords[0]) - 1
        self.end_t = end_t
        time = np.arange(0, self.end_t + 1, tau)
        self.observer_data = np.vstack((self.observer_coords[:, time], time))

        if target is None:
            self.set_target(seed=seed)
        else:
            self.target_data = np.vstack((np.array(target.coords)[:, time], time))
            self.true_params = np.array(target.get_params())

            # self.bearings = f.xy_func(
            #     self.observer_data, f.convert_to_xy(self.true_params)
            # )

            rx = self.target_data[0] - self.observer_data[0]
            ry = self.target_data[1] - self.observer_data[1]
            self.bearings = np.arctan2(ry, rx)
            self.distances = f.dist_func(self.observer_data, self.target_data)

        self.set_noise(seed=seed)
        if verbose:
            self.verbose = True
            print(
                "СКОп = {:.1f}, ".format(np.degrees(self.noise_std))
                + "tau = {}, ".format(self.tau)
                + "end_time = {}".format(end_t)
            )

    def set_target(self, p0=None, seed=None):
        if p0 is None:
            np.random.seed(seed)
            b = 0
            d = np.random.uniform(5, 50)
            c = np.random.uniform(0, 180)
            v = np.random.uniform(5, 15)
        else:
            b, d, c, v = p0

        target = Object(
            "Объект", b, d, c, v, self.observer, mode="bdcv"
        )  # создаем корабль-объект
        target.forward_movement(len(self.observer_coords[0]) - 1)
        self.true_params = target.get_params()
        time = np.arange(0, self.end_t + 1, self.tau)
        self.target_data = np.vstack((np.array(target.coords)[:, time], time))
        self.bearings = f.xy_func(self.observer_data, f.convert_to_xy(self.true_params))
        self.distances = f.dist_func(self.observer_data[0:2], self.target_data[0:2])

    def set_noise(self, seed=None):
        rng = np.random.RandomState(seed)
        self.bearings_with_noise = self.bearings.copy()
        noise = rng.normal(self.noise_mean, self.noise_std, len(self.bearings))
        self.bearings_with_noise += noise

    def mle_algorithm_v1(self, p0):
        algorithm_name = "ММП v1"

        def f(data, b, d, c, v):
            return f.xy_func(data, [b, d, c, v])

        start_time = time.perf_counter()
        res = curve_fit(
            f, self.observer_coords, self.bearings_with_noise, p0=p0, full_output=True
        )
        stop_time = time.perf_counter()
        score = res[2]["nfev"]
        if (np.diag(res[1]) < 0).any():
            perr = np.empty(4)
            perr[:] = np.nan
        else:
            perr = np.sqrt(np.diag(res[1]))

        return self.get_result(
            algorithm_name, res[0], perr, score, p0, stop_time - start_time
        )

    def mle_algorithm_v2(self, p0, verbose=False, full_output=True):
        algorithm_name = "ММП"
        start_time = time.perf_counter()
        res = lev_mar(
            f.xy_func,
            self.observer_data,
            self.bearings_with_noise,
            p0,
            verbose=verbose,
            jac=f.xy_func_jac,
            std=self.noise_std,
        )
        stop_time = time.perf_counter()
        if (np.diag(res[1]) < 0).any():
            perr = np.empty(4)
            perr[:] = np.nan
        else:
            perr = np.sqrt(np.diag(res[1]))
        if full_output:
            return self.get_result(
                algorithm_name, res[0].copy(), perr, res[2], p0, stop_time - start_time
            )
        else:
            return res[0]

    def n_bearings_algorithm(self, p0):
        algorithm_name = "Метод N пеленгов"

        start_time = time.perf_counter()
        b0 = self.bearings_with_noise[0]
        n = len(self.observer_data[0])

        bearings = self.bearings_with_noise
        bearing_origin = np.array([b0] * n)

        H = [-np.sin(bearing_origin - bearings) * 1000.0]
        H.append(np.sin(bearings) * self.observer_data[2])
        H.append(-np.cos(bearings) * self.observer_data[2])
        H = np.array(H).T
        d = self.observer_data[0] * np.sin(bearings) - self.observer_data[1] * np.cos(
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
        return self.get_result(
            algorithm_name,
            res.copy(),
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan],
            stop_time - start_time,
        )

    def real_time_process(self, p0, start_t, delta_t):
        end_t_ind = start_t // self.tau
        delta_t_ind = delta_t // self.tau
        res_t = []
        # lam = 1e-2
        for i in range(end_t_ind, self.observer_data.shape[1], delta_t_ind):
            start_time = time.perf_counter()
            res = lev_mar(
                f.xy_func,
                self.observer_data[:, : i + 1],
                self.bearings_with_noise[: i + 1],
                p0,
                verbose=False,
                jac=f.xy_func_jac,
                return_lambda=True,
                std=self.noise_std,
            )
            stop_time = time.perf_counter()
            res_t.append(
                self.get_result(
                    "ММП в реальном времени",
                    res[0],
                    np.sqrt(np.diag(res[1])),
                    res[2],
                    p0,
                    stop_time - start_time,
                )
            )
            # p0 = res[0]
            # if np.linalg.norm(p0, np.inf) > 100:
            #     p0 = [0.0, 20.0, 45.0, 10.0]
            # lam = res[3] * 4
        return res_t

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
            "ММП": self.mle_algorithm_v2,
            "Метод N пеленгов": self.n_bearings_algorithm,
        }
        algorithm = alg_dict[algorithm_name]
        if fixed_p0:
            if algorithm_name in ["ММП"]:
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
                    self.set_target(p0=target_func(seed=i))
                else:
                    self.set_target(p0=target_func())

            if not fixed_p0:
                if seeded:
                    p0 = p0_func(seed=i)
                else:
                    p0 = p0_func()
                if algorithm_name in ["ММП"]:
                    p0[0] = f.to_angle(np.radians(p0[0]))
                    p0[2] = f.to_angle(np.radians(p0[2]))
                    b, d, c, v = p0
                    p0 = [d * np.cos(b), d * np.sin(b), v * np.cos(c), v * np.sin(c)]
                else:
                    p0[0] = f.to_angle(np.radians(p0[0]))
                    p0[2] = f.to_angle(np.radians(p0[2]))

            if not fixed_noise:
                if seeded:
                    self.set_noise(seed=i)
                else:
                    self.set_noise()

            result = algorithm(p0)
            res_arr.append(result)
        if verbose:
            stop_time = time.perf_counter()
            print("Алгоритм:: " + algorithm_name)
            print(
                "Моделирование {} результатов закончено за t = {:.1f} с".format(
                    n, stop_time - start_time
                )
            )
        return res_arr

    def get_observed_information(self):
        params = f.convert_to_xy(self.last_result)
        J = f.xy_func_jac(self.observer_data, params)
        I = J.T.dot(J) / (self.noise_std ** 2)
        return I

    def get_result(self, algorithm_name, res, perr, nfev, p0, t):
        r = self.bearings_with_noise - f.xy_func(self.observer_data, res)  # residuals
        # chi_2 = sum((r) ** 2) / r.var(ddof=1)
        # r = (r + np.pi) % (2 * np.pi) - np.pi # normalization
        # if self.verbose:
        # Проверка гипотезы
        # print('z_stat = {:.2f}, p-value = {:.4f}'.format(*ztest(r, ddof=5)))
        # print('chi2_stat = {:.2f}, p-value = {:.4f}'.format(chi_2, 1 - chi2.cdf(chi_2, len(r) - 5)))
        # from matplotlib import pyplot as plt
        # plt.plot(r)
        # plt.plot(self.bearings)
        # plt.plot(f.xy_func(self.observer_data, res))
        # plt.show()

        b_end_pred = np.degrees(f.to_bearing(f.xy_func(self.observer_data[:, -1], res)))
        r_x_end = self.observer_data[0][-1] - 1000.0 * res[0] - res[2] * self.end_t
        r_y_end = self.observer_data[1][-1] - 1000.0 * res[1] - res[3] * self.end_t
        d_end_pred = np.sqrt(r_x_end ** 2 + r_y_end ** 2) / 1000.0
        res = f.convert_to_bdcv(res)

        if algorithm_name in ["ММП v1", "ММП"]:
            p0 = f.convert_to_bdcv(p0)

        self.last_result = res
        r = np.degrees(r)
        k_a = np.sum(r ** 2) / len(r)

        d = np.array([1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1])
        delta = self._get_delta(d, b_end_pred, d_end_pred)

        k_b = np.sum(delta) / 4
        k_c = []

        D = [
            [1 / 0.5, 1 / 0.05, 1 / 5, 1 / 0.05],
            [1 / 1, 1 / 0.1, 1 / 10, 1 / 0.1],
            [1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1],
            [1 / 1, 1 / 0.15, 1 / 10, 1 / 0.15],
        ]

        for d in D:
            delta = self._get_delta(d, b_end_pred, d_end_pred)
            k_c.append(int(all(delta < [1, 1, 1, 1])))

        result = {
            algorithm_name: {
                "Истинные параметры": self.true_params,
                "Полученные параметры": res,
                "Начальное приближение": p0,
                "Текущие значения": [
                    np.degrees(f.to_bearing(self.bearings[-1])),
                    self.distances[-1],
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

    def _get_data(self):
        data = {
            "Время": self.observer_data[2],
            "Истинный пеленг": np.degrees(list(map(f.to_bearing, self.bearings))),
            "Расстояние": self.distances,
            "Зашумленный пеленг": np.degrees(
                list(map(f.to_bearing, self.bearings_with_noise))
            ),
        }
        return data

    def _get_delta(self, d, b_end_pred, d_end_pred):
        b_end = np.degrees(f.to_bearing(self.bearings[-1]))
        d_end = self.distances[-1]
        d[1] /= d_end
        d[3] /= self.true_params[3]
        temp = self.true_params - self.last_result
        temp[0] = abs(b_end_pred - b_end)
        temp[1] = abs(d_end_pred - d_end)
        temp[0] = (temp[0] + 180) % 360 - 180
        temp[2] = (temp[2] + 180) % 360 - 180
        temp = abs(temp) * d
        return temp