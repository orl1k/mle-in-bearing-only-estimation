import numpy as np
from project.ship import Ship
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from project.lm import lev_mar
import time


class TMA(Ship):
    def __init__(self, observer, target=None, tau=2, mean=0, sd=np.radians(0.1), seed=None):
        np.random.seed(seed)
        self.mean = mean
        self.standart_deviation = sd
        self.observer = observer
        self.observer_coords = np.array(self.observer.coords)
        if target is not None:
            self.target = target
        else:
            self.set_target(seed)
        self.tau = tau
        self.time = np.arange(0, len(self.observer_coords[0]), self.tau)
        self.observer_data = np.vstack((self.observer_coords[:, self.time], self.time))
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
    def _xy_func2(data, params):
        x_origin, y_origin, x_velocity, y_velocity = params
        n = len(data[0])
        r_y = 1000 * y_origin + y_velocity * data[2] - data[1]
        r_x = 1000 * x_origin + x_velocity * data[2] - data[0]
        angle = np.arctan2(r_y, r_x)
        return angle

    @staticmethod
    def _xy_func2_jac(data, params):
        n = len(data[0])
        J = []
        x, y, vx, vy = params
        r_y = 1000 * y + vy * data[2] - data[1]
        r_x = 1000 * x + vx * data[2] - data[0]
        R2 = r_x ** 2 + r_y ** 2

        J.append(-1000 * r_y / R2)
        J.append(1000 * r_x / R2)
        J.append(-(data[2] * r_y) / R2)
        J.append((data[2] * r_x) / R2)

        return np.array(J).T

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
        res = lev_mar(self._b_func, self.observer_coords,
                    self.bearings_with_noise, p0, verbose=False, jac=self._b_func_jac)
        stop_time = time.perf_counter()
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0].copy(), perr, res[2], p0, stop_time - start_time)

    def mle_algorithm_v5(self, p0, verbose=False):
        algorithm_name = 'ММП v5'
        # w = np.array([np.radians(0.1)**2]*1141)
        start_time = time.perf_counter()
        res = lev_mar(self._xy_func, self.observer_coords,
                    self.bearings_with_noise, p0, verbose=False, jac=self._xy_func_jac)
        stop_time = time.perf_counter()
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0].copy(), perr, res[2], p0, stop_time - start_time)

    def mle_algorithm_v6(self, p0, verbose=False):
        algorithm_name = 'ММП v6'
        start_time = time.perf_counter()
        res = lev_mar(self._xy_func2, self.observer_data,
                    self.bearings_with_noise[self.time], p0, verbose=verbose, jac=self._xy_func2_jac, std=self.standart_deviation)
        stop_time = time.perf_counter()
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0].copy(), perr, res[2], p0, stop_time - start_time)

    def n_bearings_algorithm(self, p0):
        algorithm_name = 'Метод N пеленгов'

        start_time = time.perf_counter()
        b0 = self.bearings_with_noise[0]
        n = len(self.observer_data[0])

        bearings = self.bearings_with_noise[self.time]
        bearing_origin = np.array([b0]*n)

        H = [-np.sin(bearing_origin-bearings) * 1000]
        H.append(np.sin(bearings)*self.observer_data[2])
        H.append(-np.cos(bearings)*self.observer_data[2])
        H = np.array(H).T
        d = self.observer_data[0]*np.sin(bearings) - \
            self.observer_data[1]*np.cos(bearings)

        res = np.linalg.solve(np.dot(H.T, H),
                              np.dot(H.T, d))

        # reg = LinearRegression().fit(H, d)
        # res = reg.coef_

        res = np.insert(res, 0, b0)
        b, d = b0, res[1]
        res[0] = d * np.cos(b)
        res[1] = d * np.sin(b)

        stop_time = time.perf_counter()
        return self.get_result(algorithm_name, res.copy(), [np.nan, np.nan, np.nan, np.nan],\
             [np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan], stop_time - start_time)

    def print_verbose(self):
        # print('П0 = {}, Д0 = {} км, К = {}, V = {} м/c'.format(*self.target.get_params()))
        print('СКОп = {}, '.format(np.degrees(self.standart_deviation)) + 'tau = {}'.format(self.tau))
        # print('Предельно допустимые значения ошибок КПДО:')
        # print('- 1 градус по пеленгу,')
        # print('- 15%Д по дальности,')
        # print('- 10 градусов по курсу,')
        # print('- 10% по скорости.')

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
        plt.ylim(-1000, 35000)
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

    def swarm(self, algorithm_name='ММП v6', n=100, seeded=True, fixed_target=False, fixed_noise=False, p0_func=None, target_func=None, p0=None):
        res_arr = []

        if p0 is not None:
            fixed_p0=True
        else:
            fixed_p0=False

        if p0_func is None:
            p0_func = self.get_random_p0
        if target_func is None:
            target_func = self.get_random_p0
        
        alg_dict = {'ММП v6': self.mle_algorithm_v6, 'Метод N пеленгов': self.n_bearings_algorithm}
        algorithm = alg_dict[algorithm_name]
        if fixed_p0:
            if algorithm_name in ['ММП v6', 'ММП v5']:
                p0[0] = Ship.transform_to_angle(np.radians(p0[0]))
                p0[2] = Ship.transform_to_angle(np.radians(p0[2]))
                b, d, c, v = p0
                p0 = [d * np.cos(b), d * np.sin(b), v * np.cos(c), v * np.sin(c)]
            else:
                p0[0] = Ship.transform_to_angle(np.radians(p0[0]))
                p0[2] = Ship.transform_to_angle(np.radians(p0[2]))
        for i in range(n):
            
            if not fixed_target:
                if seeded:
                    self.set_target(p0=target_func(seed=i))
                else:
                    self.set_target(p0=target_func())

            if not fixed_p0:
                if seeded:
                    p0 = p0_func(seed = i)
                else:
                    p0 = p0_func()
                if algorithm_name in ['ММП v6', 'ММП v5']:
                    p0[0] = Ship.transform_to_angle(np.radians(p0[0]))
                    p0[2] = Ship.transform_to_angle(np.radians(p0[2]))
                    b, d, c, v = p0
                    p0 = [d * np.cos(b), d * np.sin(b), v * np.cos(c), v * np.sin(c)]
                else:
                    p0[0] = Ship.transform_to_angle(np.radians(p0[0]))
                    p0[2] = Ship.transform_to_angle(np.radians(p0[2]))

            if not fixed_noise:
                if seeded:
                    self.set_noise(seed=i)
                else: 
                    self.set_noise()
            
            result = algorithm(p0)
            res_arr.append(result)
        return res_arr

    @staticmethod
    def get_random_p0(seed = None):
        if seed is not None:
            np.random.seed(seed)
        b = 0
        d = np.random.uniform(5, 50)
        c = np.random.uniform(0, 180)
        v = np.random.uniform(5, 15)
        return [b, d, c, v]

    def linear(self):
        self.mu = np.mean(self.bearings_with_noise)
        self.shifted_bearings_with_noise = self.bearings_with_noise - self.mu
        plt.plot(np.tan(self.shifted_bearings_with_noise))
        plt.show()

    def get_observed_information(self):
        params = Ship.convert_to_xy(self.last_result)
        # f = self._xy_func2(self.observer_data, params)
        # y = self.bearings_with_noise[self.time]
        # l = np.sum(np.log(stats.norm.pdf(f, y, np.radians(0.5))))
        J = self._xy_func2_jac(self.observer_data, params)
        I = J.T.dot(J) / (self.standart_deviation ** 2)
        from matplotlib import pyplot as plt
        return I
        
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

        elif algorithm_name in ['ММП v6']:
            r = np.degrees(self.bearings_with_noise[self.time] -
                         self._xy_func2(self.observer_data, res)) # residuals
            res = Ship.convert_to_bdcv(res)
            p0 = Ship.convert_to_bdcv(p0)

        elif algorithm_name in ['Метод N пеленгов']:
            r = np.degrees(self.bearings_with_noise[self.time] -
                         self._xy_func2(self.observer_data, res))
            res = Ship.convert_to_bdcv(res)
        
        else:
            r = np.degrees(self.bearings_with_noise -
                         self._xy_func(self.observer_coords, res)) # residuals
            res = Ship.convert_to_bdcv(res)
            p0 = Ship.convert_to_bdcv(p0)
        
        self.last_result = res
        k_a = np.sum(r ** 2) / len(r)
        
        d = np.array([1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1])
        delta = self._get_delta(d)

        k_b = np.sum(delta) / 4
        k_c = []

        D = [[1 / 0.5, 1 / 0.05, 1 / 5, 1 / 0.05], [1 / 1, 1 / 0.1, 1 / 10, 1 / 0.1],\
             [1 / 1, 1 / 0.15, 1 / 10, 1 / 0.1], [1 / 1, 1 / 0.15, 1 / 10, 1 / 0.15]]

        for d in D:
            delta = self._get_delta(d)
            k_c.append(int(all(delta < [1, 1, 1, 1])))

        result = {algorithm_name: {'Истинные параметры': self.true_params,
                                   'Полученные параметры': res,
                                   'Начальное приближение': p0,
                                   'СКО параметров': perr,
                                   'Ка, Кб, Кс': [k_a, k_b, k_c],
                                   'Время работы': [t],
                                   'Число вычислений функции, число итераций': nfev
                                   }}
        return result

    def _get_data(self):
        data = {'Время': self.time,
                'Истинный пеленг': np.degrees(list(map(Ship.transform_to_bearing, self.bearings)))[self.time],
                'Расстояние': self.distance_array[self.time],
                'Зашумленный пеленг': np.degrees(list(map(Ship.transform_to_bearing, self.bearings_with_noise)))[self.time],
                }
        return data

    def _get_delta(self, d):
        d[1] /= self.true_params[1]
        d[3] /= self.true_params[3]
        temp = self.true_params - self.last_result
        temp[0] = (temp[0] + 180) % 360 - 180
        temp[2] = (temp[2] + 180) % 360 - 180
        temp = abs(temp) * d
        return temp