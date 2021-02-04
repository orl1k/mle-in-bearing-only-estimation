import numpy as np
from ship import Ship
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class TMA(Ship):
    def __init__(self, observer, target, mean=0, standart_deviation=0.3 * np.pi / 180):
        self.mean = mean
        self.standart_deviation = standart_deviation
        self.observer = observer
        self.target = target
        self.set_bearings_and_distances()
        self.set_noise()

    @staticmethod
    def func_v1(data, x_origin, y_origin, x_velocity, y_velocity):
        n = len(observer.coords[0])
        num = 1000 * y_origin * np.array([1]*n) + y_velocity * \
            np.array(range(n)) - data[1]
        den = 1000 * x_origin * np.array([1]*n) + x_velocity * \
            np.array(range(n)) - data[0]
        angle = np.arctan2(num, den)
        return angle

    @staticmethod
    def func_v2(data, distance, x_velocity, y_velocity):
        n = len(data[0])
        num = 1000 * distance * np.sin(data[2]) + y_velocity * \
            np.array(range(n)) - data[1]
        den = 1000 * distance * np.cos(data[2]) + x_velocity * \
            np.array(range(n)) - data[0]
        angle = np.arctan2(num, den)

        return angle

    @staticmethod
    def func_v3(data, bearing, distance, course, velocity):
        n = len(data[0])
        x_velocity = velocity * np.cos(course)
        y_velocity = velocity * np.sin(course)
        num = 1000 * distance * np.sin(bearing) + y_velocity * \
            np.array(range(n)) - data[1]
        den = 1000 * distance * np.cos(bearing) + x_velocity * \
            np.array(range(n)) - data[0]
        angle = np.arctan2(num, den)

        return angle

    def func_v4(self, data, bearing, distance, course, velocity):
        n = len(data[0])
        x_velocity = velocity * np.cos(course)
        y_velocity = velocity * np.sin(course)
        num = 1000 * distance * np.sin(bearing) + y_velocity * \
            np.array(range(n)) - data[1]
        den = 1000 * distance * np.cos(bearing) + x_velocity * \
            np.array(range(n)) - data[0]
        angle = np.arctan2(num, den)

        return angle - self.mu

    def bearing_function(self, args):
        observer_x, observer_y, target_x, target_y = args
        num = target_y - observer_y
        den = target_x - observer_x
        angle = (np.arctan2(num, den))
        return angle

    def set_bearings_and_distances(self):
        bearing_array = []
        distance_array = []
        eps = 10
        for i in range(len(self.observer.coords[0])):
            bearing = self.bearing_function([self.observer.coords[0][i],
                                             self.observer.coords[1][i],
                                             self.target.coords[0][i],
                                             self.target.coords[1][i]])
            x = self.observer.coords[0][i] - self.target.coords[0][i]
            y = self.observer.coords[1][i] - self.target.coords[1][i]
            r_vector = np.array((x, y))
            distance = np.linalg.norm(r_vector)
            distance_array.append(distance)
            if distance < eps:
                try:
                    bearing_array.append(bearing_array[i-1])
                except(IndexError):
                    bearing_array.append(bearing)
                    print(
                        'Корабли находятся слишком близко друг к другу в начальный момент времени')
            else:
                bearing_array.append(bearing)
        self.bearings = np.array(bearing_array)
        self.distance_array = np.array(distance_array)

    def set_target(self, *args):
        if len(args[0]) != 4:
            target_bearing = 0
            target_distance = np.random.uniform(0, 45)
            target_course = np.random.uniform(0, 180)
            target_velocity = np.random.uniform(5, 15)
        else:
            target_bearing, target_distance, target_course, target_velocity = args[0]
        self.target = Ship('объект', target_bearing, target_distance, target_course,
                           target_velocity, observer, mode='bdcv')  # создаем корабль-объект
        self.target.forward_movement(len(observer.coords[0]) - 1)
        self.set_bearings_and_distances()
        self.set_noise()

    def set_noise(self):
        self.bearings_with_noise = self.bearings.copy()
        noise = np.random.normal(
            self.mean, self.standart_deviation, len(self.bearings))
        self.bearings_with_noise += noise

    def mle_algorithm_v1(self, p0):
        algorithm_name = 'ММП v1'
        res = curve_fit(self.func_v1, observer.coords,
                        self.bearings_with_noise, p0=p0, full_output=True)
        bearing, distance = Ship.convert_to_polar(res[0][0], res[0][1])
        course, velocity = Ship.convert_to_polar(res[0][2], res[0][3])
        score = res[2]['nfev']
        print(res[0])
        return self.get_result(algorithm_name, [bearing, distance, course, velocity], score, p0)

    def mle_algorithm_v2(self, b0, p0):
        algorithm_name = 'ММП v2'
        bearing_origin = [b0]*len(self.observer.coords[0])
        data = self.observer.coords
        data.append(bearing_origin)
        popt, pcov = curve_fit(self.func_v2, data,
                               self.bearings_with_noise, p0=p0, )
        course, velocity = Ship.convert_to_polar(popt[1], popt[2])
        return self.get_result(algorithm_name, [b0, popt[0], course, velocity])

    def mle_algorithm_v3(self, p0):
        algorithm_name = 'ММП v3'
        res = curve_fit(self.func_v3, self.observer.coords,
                        self.bearings_with_noise, p0=p0, full_output=True, maxfev=10000)
        score = res[2]['nfev']
        self.last_result = res[0].copy()
        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0], perr, score, p0)

    def mle_algorithm_v32(self, p0):
        algorithm_name = 'ММП v32'
        self.mu = np.mean(self.bearings_with_noise)
        res = curve_fit(self.func_v4, self.observer.coords,
                        self.bearings_with_noise - self.mu, p0=p0, full_output=True, maxfev=10000)
        score = res[2]['nfev']
        self.last_result = res[0].copy()

        perr = np.sqrt(np.diag(res[1]))
        return self.get_result(algorithm_name, res[0], perr, score, p0)

    def mle_algorithm_v33(self, p0):
        algorithm_name = 'ММП v3'
        res, cov = curve_fit(self.func_v3, observer.coords,
                             self.bearings_with_noise, p0=p0, bounds=((np.pi * 85 / 180, -np.pi, -np.inf, 1), (np.pi * 95 / 180, 35, np.pi, 20)))
        self.last_result = res.copy()
        return self.get_result(algorithm_name, res, cov, p0)

    def n_bearings_algorithm(self, b0):
        algorithm_name = 'Метод N пеленгов'
        n = len(observer.coords[0])

        bearing_origin = np.array([b0]*n)

        H = [-np.sin(bearing_origin - self.bearings_with_noise)*1000]
        H.append(np.sin(self.bearings_with_noise)*np.array(range(n)))
        H.append(-np.cos(self.bearings_with_noise)*np.array(range(n)))
        H = np.array(H).transpose()
        d = observer.coords[0]*np.sin(self.bearings_with_noise) - \
            observer.coords[1]*np.cos(self.bearings_with_noise)

        res = np.linalg.solve(np.dot(H.transpose(), H),
                              np.dot(H.transpose(), d))
        course, velocity = Ship.convert_to_polar(res[1], res[2])

        # course, velocity = Ship.convert_to_polar(skm.coef_[1], skm.coef_[2])
        return self.get_result(algorithm_name, [b0, res[0], course, velocity], [1, 1, 1, 1], 10, [1, 1, 1, 1])

    def plot_trajectories(self):
        plt.plot(self.observer.coords[0], self.observer.coords[1])
        plt.plot(self.target.coords[0], self.target.coords[1])
        m = len(self.observer.coords[0]) // 2
        plt.arrow(self.observer.coords[0][m-30], self.observer.coords[1][m-30], self.observer.coords[0][m+1] - self.observer.coords[0][m],
                  self.observer.coords[1][m+1] - self.observer.coords[1][m], shape='full', lw=0, head_width=300, head_starts_at_zero=True)
        plt.arrow(self.target.coords[0][m-30], self.target.coords[1][m-30], self.target.coords[0][m+1] - self.target.coords[0][m],
                  self.target.coords[1][m+1] - self.target.coords[1][m], shape='full', lw=0, head_width=300, head_starts_at_zero=True, color='#ff7f0e')
        plt.axis('square')
        plt.xlim(-5000, 10000)
        plt.ylim(0, 40000)
        plt.grid()
        ax = plt.gca()
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
        b, d, c, v = self.last_result
        plt.plot(np.degrees([Ship.transform_to_bearing(i)
                             for i in self.func_v3(self.observer.coords, b, d, c, v)]))
        ax = plt.gca()
        ax.set_xlabel('время (с)')
        ax.set_ylabel('пеленг (г)')
        ax.legend(['Истинный пеленг', 'Расчетный пеленг'])
        plt.xlim([0, len(self.bearings) - 1])
        plt.show()

    def plot_tan(self):
        plt.plot(np.tan(self.bearings), linewidth=5.0)
        b, d, c, v = self.last_result
        plt.plot(np.tan(self.func_v3(self.observer.coords, b, d, c, v)))
        ax = plt.gca()
        ax.set_xlabel('время (с)')
        ax.set_ylabel('бета (г)')
        ax.legend(['Истинный бета', 'Расчетный бета'])
        plt.xlim([0, len(self.bearings) - 1])
        plt.show()

    def plot_rad(self):
        plt.plot(self.bearings, linewidth=5.0)
        b, d, c, v = self.last_result
        plt.plot(self.func_v3(self.observer.coords, b, d, c, v))
        ax = plt.gca()
        ax.set_xlabel('время (с)')
        ax.set_ylabel('бета (г)')
        ax.legend(['Истинный бета', 'Расчетный бета'])
        plt.xlim([0, len(self.bearings) - 1])
        plt.show()

    def plot_distances(self):
        plt.plot(self.distance_array, linewidth=5.0)
        d = [np.array(self.observer.coords[0]) -
             np.array(self.target.coords[0])]
        d.append(
            np.array(self.observer.coords[1]) - np.array(self.target.coords[1]))
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
                J[i][j] = sum((self.bearings_with_noise - self.func_v1(
                    self.observer.coords, xlist[i], ylist[j], 5.30330086, 5.30330086))**2)
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
        # clist = np.linspace(-np.pi, np.pi, 50)
        # vlist = np.linspace(-20, 20.0, 50)
        # C, V = np.meshgrid(clist, vlist)
        # J = np.zeros((len(clist), len(vlist)))
        # print(sum((self.bearing_array_with_noise - self.func_v3(self.observer.coords, np.pi/2, 10, np.pi/4, 7.5))**2))
        # for i in range(len(clist)):
        #     for j in range(len(vlist)):
        #         J[i][j] = sum((self.bearing_array_with_noise - self.func_v3(self.observer.coords, np.pi/2, 10, clist[i], vlist[j]))**2)
        # fig, ax = plt.subplots()
        # lev = [0.0000001, 1]
        # ax.contour(C, V, J, lev)
        # plt.show()

        # blist = np.linspace(-np.pi, np.pi, 50)
        # dlist = np.linspace(-100.0, 100.0, 50)
        # B, D = np.meshgrid(blist, dlist)
        # J = np.zeros((len(blist), len(dlist)))
        # print(sum((self.bearing_array_with_noise - self.func_v3(self.observer.coords, 1, 1, np.pi/4, 7.5))**2))
        # for i in range(len(blist)):
        #     for j in range(len(dlist)):
        #         J[i][j] = sum((self.bearing_array_with_noise - self.func_v3(self.observer.coords, blist[i], dlist[j], np.pi/4, 7.5))**2)
        # fig, ax = plt.subplots()
        # lev = [10, 50, 100, 200, 400, 800, 1600]
        # ax.contour(B, D, J, lev)
        # plt.show()

    def get_data(self):
        data = {'Время': range(len(self.bearings)),
                'Истинный пеленг': np.degrees([Ship.transform_to_bearing(i) for i in self.bearings]),
                'Расстояние': self.distance_array / 1000,
                'Зашумленный пеленг': np.degrees([Ship.transform_to_bearing(i) for i in self.bearings_with_noise]),
                }
        return data

    def get_fim(self):
        I = np.zeros(4)
        I[1, 1] = np.sin(self.bearings)**2
        pass

    def swarm(self, n):
        r_arr = []
        for i in range(n):
            b0 = 0
            d0 = np.random.uniform(5, 40)
            c0 = np.random.uniform(0, 180)
            v0 = np.random.uniform(5, 15)
            self.set_target([b0, d0, c0, v0])
            b0 = Ship.transform_to_angle(np.radians(b0))
            c0 = Ship.transform_to_angle(np.radians(c0))
            p0 = [b0, d0, c0, v0]
            # p0 = [np.pi / 2, 20.0, np.pi / 4, 10.0]
            try:
                result = self.mle_algorithm_v3(p0)
                r_arr.append(result)
                self.set_noise()
            except(RuntimeError):
                print('runtime error')
                print(p0)
        return r_arr

    def linear(self):
        self.mu = np.mean(self.bearings_with_noise)
        self.shifted_bearings_with_noise = self.bearings_with_noise - self.mu
        plt.plot(np.tan(self.shifted_bearings_with_noise))
        plt.show()

    def get_result(self, algorithm_name, params, perr, nfev, p0):

        # f = self.func_v3(self.observer.coords, params[0], params[1], params[2], params[3])
        # bwn = self.bearings_with_noise
        # ss_res = sum((np.array(f) - bwn)**2)
        # ss_tol = sum((bwn - np.mean(bwn))**2)
        # R2 = 1 - ss_res / ss_tol

        params[0] %= 2 * np.pi
        params[2] %= 2 * np.pi

        if params[3] < 0:
            params[3] = -params[3]
            params[2] = params[2] - np.pi

        if params[1] < 0:
            params[1] = -params[1]
            params[0] = params[0] - np.pi

        arr = np.degrees(self.bearings_with_noise -
                         self.func_v3(self.observer.coords,
                                      params[0], params[1], params[2], params[3]))
        k_a = np.dot(arr, arr.transpose()) / len(arr)

        true_params = self.target.get_params()

        params[0] = np.degrees(Ship.transform_to_bearing(params[0]))
        params[2] = np.degrees(Ship.transform_to_bearing(params[2]))
        p0[0] = np.degrees(Ship.transform_to_bearing(p0[0]))
        p0[2] = np.degrees(Ship.transform_to_bearing(p0[2]))

        temp = true_params - np.array(params)
        temp[0] = (temp[0] + 180) % 360 - 180
        temp[2] = (temp[2] + 180) % 360 - 180

        d = np.array([1 / 1, 1 / (0.15 * true_params[1]),
                      1 / 10, 1 / (0.1 * true_params[3])])
        temp = abs(temp) * d
        k_b = np.sum(temp) / 4

        perr[0] = np.degrees(perr[0])
        perr[2] = np.degrees(perr[2])

        result = {algorithm_name: {'Истинные параметры': self.target.get_params(),
                                   'Полученные параметры': list(params),
                                   'Начальное приближение': p0,
                                   'Оценка': [k_a, k_b],
                                   'Число вычислений функции': [nfev],
                                   'Среднеквадратичное отклонение параметров': perr,
                                   'Данные': self.get_data()}}

        return result
