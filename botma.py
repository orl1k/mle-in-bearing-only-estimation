import numpy as np
from ship import Ship
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import lm
import time


class TMA(Ship):
    def __init__(self, observer, target, mean=0, sd=np.radians(0.1), seed=None):
        np.random.seed(seed)
        self.mean = mean
        self.standart_deviation = sd
        self.observer = observer
        self.target = target
        self.observer_coords = np.array(self.observer.coords)
        self._set_bearings_and_distances()
        self.set_noise()
        self.true_params = self.target.get_params()
        # self.true_params[0] = self.transform_to_angle(
        #     np.radians(self.true_params[0]))
        # self.true_params[2] = self.transform_to_angle(
        #     np.radians(self.true_params[2]))

    @staticmethod
    def _b_func(data, params):
        bearing, distance, course, velocity = params
        n = len(data[0])
        x_velocity = velocity * np.cos(course)
        y_velocity = velocity * np.sin(course)
        num = 1000 * distance * np.sin(bearing) + y_velocity * \
            np.array(range(n)) - data[1]
        den = 1000 * distance * np.cos(bearing) + x_velocity * \
            np.array(range(n)) - data[0]
        angle = np.arctan2(num, den)

        return angle

    @staticmethod
    def _b_func_jac(data, params):
        J = []
        b, d, c, v = params
        n = len(data[0])

        vx = v * np.cos(c)
        vy = v * np.sin(c)
        x = d * np.cos(b)
        y = d * np.sin(b)

        r_y = 1000 * y + vy * np.arange(n) - data[1]
        r_x = 1000 * x + vx * np.arange(n) - data[0]
        R2 = r_x ** 2 + r_y ** 2

        J.append((1000 * (x * r_x + y * r_y)) / R2)
        J.append((1000 * (np.sin(b) * r_x - np.cos(b) * x * r_y)) / R2)
        J.append((vx * r_x + vy * r_y) * np.arange(n) / R2)
        J.append((np.sin(c) * r_x - np.cos(c) * r_y) * np.arange(n) / R2)

        return np.array(J).T

    @staticmethod
    def _b_func2(data, params):
        bearing, distance, course, velocity = params
        x_velocity = velocity * np.cos(course)
        y_velocity = velocity * np.sin(course)
        d = data.T
        num = 1000 * distance * np.sin(bearing) + y_velocity * \
            d[2] - d[1]
        den = 1000 * distance * np.cos(bearing) + x_velocity * \
            d[2] - d[0]
        angle = np.arctan2(num, den)

        return angle.reshape(data.shape[0], 1)

    @staticmethod
    def _xy_func(data, params):
        x_origin, y_origin, x_velocity, y_velocity = params
        n = len(data[0])
        r_y = 1000 * y_origin + y_velocity * np.arange(n) - data[1]
        r_x = 1000 * x_origin + x_velocity * np.arange(n) - data[0]
        angle = np.arctan2(r_y, r_x)
        return angle

    @staticmethod
    def _xy_func_jac(data, params):
        n = len(data[0])
        J = []
        x, y, vx, vy = params
        r_y = 1000 * y + vy * np.arange(n) - data[1]
        r_x = 1000 * x + vx * np.arange(n) - data[0]
        R2 = r_x ** 2 + r_y ** 2

        J.append(-1000 * r_y / R2)
        J.append(1000 * r_x / R2)
        J.append(-(np.arange(n) * r_y) / R2)
        J.append((np.arange(n) * r_x) / R2)

        return np.array(J).T

    def _normalized_b_func(self, data, params):
        bearing, distance, course, velocity = params
        n = len(data[0])
        x_velocity = velocity * np.cos(course)
        y_velocity = velocity * np.sin(course)
        num = 1000 * distance * np.sin(bearing) + y_velocity * \
            np.array(range(n)) - data[1]
        den = 1000 * distance * np.cos(bearing) + x_velocity * \
            np.array(range(n)) - data[0]
        angle = np.arctan2(num, den)

        return angle - self.mu

    def _bearing_function(self, params):
        observer_x, observer_y, target_x, target_y = params
        num = target_y - observer_y
        den = target_x - observer_x
        angle = (np.arctan2(num, den))
        return angle

    def _set_bearings_and_distances(self):
        bearing_array = []
        distance_array = []
        eps = 10
        for i in range(len(self.observer_coords[0])):
            params = [self.observer_coords[0][i], self.observer_coords[1][i],
                      self.target.coords[0][i], self.target.coords[1][i]]
            bearing = self._bearing_function(params)
            x = self.observer_coords[0][i] - self.target.coords[0][i]
            y = self.observer_coords[1][i] - self.target.coords[1][i]
            r_vector = np.array((x, y))
            distance = np.linalg.norm(r_vector)
            distance_array.append(distance)
            if distance < eps:
                try:
                    bearing_array.append(bearing_array[i-1])
                except(IndexError):
                    bearing_array.append(bearing)
                    print(
                        'Объекты находятся слишком близко друг к другу в начальный момент времени')
            else:
                bearing_array.append(bearing)
        self.bearings = np.array(bearing_array)
        self.distance_array = np.array(distance_array) / 1000

    def set_target(self, p0 = None, seed = None):
        if p0 is None:
            np.random.seed(seed)
            b = 0
            d = np.random.uniform(5, 50)
            c = np.random.uniform(0, 180)
            v = np.random.uniform(5, 15)

        else:
            b, d, c, v = p0
        self.target = Ship('Объект', b, d, c,
                           v, self.observer, mode='bdcv')  # создаем корабль-объект
        self.target.forward_movement(len(self.observer_coords[0]) - 1)
        self._set_bearings_and_distances()
        self.true_params = self.target.get_params()
        self.set_noise(seed = seed)

    def set_noise(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
        self.bearings_with_noise = self.bearings.copy()
        noise = np.random.normal(
            self.mean, self.standart_deviation, len(self.bearings))
        self.bearings_with_noise += noise

    def mle_algorithm_v1(self, p0):
        algorithm_name = 'ММП v1'
        def f(data, x, y, vx, vy): return self._xy_func(data, [x, y, vx, vy])
        start_time = time.perf_counter()
        res = curve_fit(f, self.observer_coords,
                        self.bearings_with_noise, p0=p0, full_output=True)
        stop_time = time.perf_counter()
        bearing, distance = Ship.convert_to_polar(res[0][0], res[0][1])
        course, velocity = Ship.convert_to_polar(res[0][2], res[0][3])
        r = [bearing, distance, course, velocity]
        score = res[2]['nfev']
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, r, perr, score, p0, stop_time - start_time)

    def mle_algorithm_v2(self, p0):
        algorithm_name = 'ММП v2'
        def f(data, b, d, c, v): return self._b_func(data, [b, d, c, v])
        start_time = time.perf_counter()
        res = curve_fit(f, self.observer_coords,
                        self.bearings_with_noise, p0=p0, full_output=True)
        stop_time = time.perf_counter()
        score = res[2]['nfev']
        score = res[2]['nfev']
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0], perr, score, p0, stop_time - start_time)

    def mle_algorithm_v3(self, p0):
        algorithm_name = 'ММП v3'
        self.mu = np.mean(self.bearings_with_noise)
        res = curve_fit(self._normalized_b_func, self.observer_coords,
                        self.bearings_with_noise - self.mu, p0=p0, full_output=True, maxfev=10000)
        score = res[2]['nfev']
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0], perr, score, p0)

    def mle_algorithm_v4(self, p0):
        algorithm_name = 'ММП v4'
        # w = np.array([np.radians(0.1)**2]*1141)
        start_time = time.perf_counter()
        res = lm.lm(self._b_func, self.observer_coords,
                    self.bearings_with_noise, p0, verbose=False, jac=self._b_func_jac)
        stop_time = time.perf_counter()
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0].copy(), perr, res[2], p0, stop_time - start_time)

    def mle_algorithm_v5(self, p0):
        algorithm_name = 'ММП v5'
        # w = np.array([np.radians(0.1)**2]*1141)
        start_time = time.perf_counter()
        res = lm.lm(self._xy_func, self.observer_coords,
                    self.bearings_with_noise, p0, verbose=False, jac=self._xy_func_jac)
        stop_time = time.perf_counter()
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0].copy(), perr, res[2], p0, stop_time - start_time)

    def n_bearings_algorithm(self):
        algorithm_name = 'Метод N пеленгов'

        start_time = time.perf_counter()
        b0 = self.bearings_with_noise[0]
        n = len(self.observer_coords[0])

        bearing_origin = np.array([b0]*n)

        H = [-np.sin(bearing_origin - self.bearings_with_noise)*1000]
        H.append(np.sin(self.bearings_with_noise)*np.array(range(n)))
        H.append(-np.cos(self.bearings_with_noise)*np.array(range(n)))
        H = np.array(H).T
        d = self.observer_coords[0]*np.sin(self.bearings_with_noise) - \
            self.observer_coords[1]*np.cos(self.bearings_with_noise)

        res = np.linalg.solve(np.dot(H.T, H),
                              np.dot(H.T, d))

        # reg = LinearRegression().fit(H, d)
        # res = reg.coef_

        res = np.insert(res, 0, b0)
        res[2], res[3] = Ship.convert_to_polar(res[2], res[3])
        self.last_result = res.copy()

        stop_time = time.perf_counter()
        return self.get_result(algorithm_name, res.copy(), [0, 0, 0, 0], [0], [0, 0, 0, 0], stop_time - start_time)

    def plot_trajectories(self):
        plt.plot(self.observer_coords[0], self.observer_coords[1])
        plt.plot(self.target.coords[0], self.target.coords[1])
        m = len(self.observer_coords[0]) // 2
        plt.arrow(self.observer_coords[0][m-30], self.observer_coords[1][m-30], self.observer_coords[0][m+1] - self.observer_coords[0][m],
                  self.observer_coords[1][m+1] - self.observer_coords[1][m], shape='full', lw=0, head_width=300, head_starts_at_zero=True)
        plt.arrow(self.target.coords[0][m-30], self.target.coords[1][m-30], self.target.coords[0][m+1] - self.target.coords[0][m],
                  self.target.coords[1][m+1] - self.target.coords[1][m], shape='full', lw=0, head_width=300, head_starts_at_zero=True, color='#ff7f0e')
        plt.axis('square')
        plt.xlim(-5000, 10000)
        plt.ylim(0, 45000)
        plt.grid()
        ax = plt.gca()
        # if self.last_result is not None:
        #     from matplotlib.patches import Ellipse
        #     [b, d] = self.last_result[:2]
        #     [x, y] = [1000*d*np.cos(b), 1000*d*np.sin(b)]
        #     cov = self.last_cov
        #     eig = np.linalg.eig(cov)
        #     angle = np.arctan2(max(eig)/min(eig))
        #     confid = Ellipse([x, y], 1000, 1000)
        #     ax.add_artist(confid)
        ax.legend(['Наблюдатель', 'Объект'])
        plt.show()

    def plot_algorithm_comparsion(self):
        # fig, ax = plt.subplots()
        # ax.plot(np.degrees(self.bearing_array_with_noise),
        #         label='зашумленный пеленг')
        # ax.plot(np.degrees(self.bearing_array),
        #         label='истинный пеленг', linewidth=4, color='tab:blue')
        # plt.legend()
        # ax = plt.gca()
        # plt.xlim([0, len(self.bearing_array) - 1])
        # ax.set_xlabel('время (с)')
        # ax.set_ylabel('пеленг (г)')
        # plt.show()
        pass

    def plot_bearings(self):
        plt.plot(np.degrees([Ship.transform_to_bearing(i)
                             for i in self.bearings]), linewidth=5.0)
        plt.plot(np.degrees([Ship.transform_to_bearing(i)
                             for i in self._b_func(self.observer_coords, self.last_result)]))
        ax = plt.gca()
        ax.set_xlabel('время (с)')
        ax.set_ylabel('пеленг (г)')
        ax.legend(['Истинный пеленг', 'Расчетный пеленг'])
        plt.xlim([0, len(self.bearings) - 1])
        plt.show()

    def plot_distances(self):
        plt.plot(self.distance_array, linewidth=5.0)
        d = [np.array(self.observer_coords[0]) -
             np.array(self.target.coords[0])]
        d.append(
            np.array(self.observer_coords[1]) - np.array(self.target.coords[1]))
        plt.plot(np.linalg.norm(np.array(d), axis=0))
        ax = plt.gca()
        ax.set_xlabel('время (с)')
        ax.set_ylabel('дистанция (м)')
        ax.legend(['Истинное расстояние', 'Расчетное расстояние'])
        plt.xlim([0, len(self.bearings) - 1])
        plt.show()

    def plot_contour_lines(self):
        n = 50
        xlist = np.linspace(-0.1, 0.1, n)
        ylist = np.linspace(9.9, 10.1, n)
        X, Y = np.meshgrid(xlist, ylist)
        J = np.zeros((n, n))
        for i in range(len(xlist)):
            for j in range(len(ylist)):
                J[i][j] = sum((self.bearings_with_noise - self._xy_func(
                    self.observer_coords, xlist[i], ylist[j], 5.30330086, 5.30330086))**2)
        # fig, ax = plt.subplots()
        # lev = [0.5, 10, 20, 40, 80, 100, 200, 500]
        # ax.contour(C, V, J, lev)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, J, cmap="autumn_r", lw=0.5, rstride=1, cstride=1)
        ax.contour(X, Y, J, 10, lw=3, cmap="autumn_r",
                   linestyles="solid", offset=-1)
        ax.contour(X, Y, J, 10, lw=3, colors="k", linestyles="solid")
        plt.show()

    def swarm(self, n):
        res_arr = []
        for i in range(n):
            # p0 = [np.pi / 2, 10.0, np.pi / 4, 10.0]
            self.set_target(seed = i)
            p0 = self.get_random_p0(seed = i + 1000)
            [b, d, c, v] = p0
            p0 = [d * np.cos(b), d * np.sin(b), v * np.cos(c), v * np.sin(c)]
            try:
                # result = self.n_bearings_algorithm()
                # result = self.mle_algorithm_v2(p0)
                result = self.mle_algorithm_v5(p0)
                res_arr.append(result)
            except(RuntimeError):
                print('Runtime error')
        return res_arr

    def get_random_p0(self, seed = None):
        if seed is not None:
            np.random.seed(seed)
        b = 0
        d = np.random.uniform(5, 50)
        c = np.random.uniform(0, 180)
        v = np.random.uniform(5, 15)
        b = Ship.transform_to_angle(np.radians(b))
        c = Ship.transform_to_angle(np.radians(c))
        return [b, d, c, v]

    def linear(self):
        self.mu = np.mean(self.bearings_with_noise)
        self.shifted_bearings_with_noise = self.bearings_with_noise - self.mu
        plt.plot(np.tan(self.shifted_bearings_with_noise))
        plt.show()

    def get_result(self, algorithm_name, res, perr, nfev, p0, t):

        # f = self.b_func(self.observer_coords, params)
        # bwn = self.bearings_with_noise
        # ss_res = sum((np.array(f) - bwn)**2)
        # ss_tol = sum((bwn - np.mean(bwn))**2)
        # R2 = 1 - ss_res / ss_tol
        
        if algorithm_name in ['ММП v4', 'ММП v2']:
            r = np.degrees(self.bearings_with_noise -
                         self._b_func(self.observer_coords, res)) # residuals

            res[0] %= 2 * np.pi
            res[2] %= 2 * np.pi

            if res[3] < 0:
                res[3] = -res[3]
                res[2] = res[2] - np.pi

            if res[1] < 0:
                res[1] = -res[1]
                res[0] = res[0] - np.pi

            res[0] = np.degrees(Ship.transform_to_bearing(res[0]))
            res[2] = np.degrees(Ship.transform_to_bearing(res[2]))

            p0[0] = np.degrees(Ship.transform_to_bearing(p0[0]))
            p0[2] = np.degrees(Ship.transform_to_bearing(p0[2]))

            perr[0] = np.degrees(perr[0])
            perr[2] = np.degrees(perr[2])

        else:
            r = np.degrees(self.bearings_with_noise -
                         self._xy_func(self.observer_coords, res)) # residuals
            res = Ship.convert_to_bdcv(res)
            # perr = Ship.convert_to_bdcv(perr)
            p0 = Ship.convert_to_bdcv(p0)
            
        self.last_result = res
        k_a = np.sum(r ** 2) / len(r)

        temp = self.true_params - np.array(res)
        temp[0] = (temp[0] + 180) % 360 - 180
        temp[2] = (temp[2] + 180) % 360 - 180

        d = np.array([1 / 1, 1 / (self.true_params[1] * 0.15),
                      1 / 10, 1 / (self.true_params[3] * 0.1)])
        temp = abs(temp) * d
        k_b = np.sum(temp) / 4

        k_c = int(all(temp < [1, 1, 1, 1]))

        result = {algorithm_name: {'Истинные параметры': self.true_params,
                                   'Полученные параметры': list(res),
                                   'Начальное приближение': p0,
                                   'Оценка': [k_a, k_b, k_c],
                                   'Число вычислений функции': nfev,
                                   'Среднеквадратичное отклонение параметров': perr,
                                   'Время работы': [t],
                                   'Данные': self._get_data()}}

        return result

    def _get_data(self):
        data = {'Время': range(len(self.bearings)),
                'Истинный пеленг': np.degrees([Ship.transform_to_bearing(i) for i in self.bearings]),
                'Расстояние': self.distance_array,
                'Зашумленный пеленг': np.degrees([Ship.transform_to_bearing(i) for i in self.bearings_with_noise]),
                }
        return data
