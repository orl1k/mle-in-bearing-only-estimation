import numpy as np
from tma.object import Object
import tma.functions as f


class Model():
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
        self.seed = seed
        if end_t is None:
            self.end_t = len(self.observer.coords[0]) - 1
        else:
            self.end_t = end_t
        if target == None:
            self.new_target(seed=seed)
        else:
            self.target = target
            self.set_data()
            self.set_bearings()
            self.set_noise()
        if verbose:
            self.verbose = True
            print(
                "СКОп = {:.1f}, ".format(np.degrees(self.noise_std))
                + "tau = {}, ".format(self.tau)
                + "end_time = {}".format(self.end_t)
            )

    def new_target(self, p0=None, seed=None):
        if p0 is None:
            np.random.seed(seed)
            b = 0
            d = np.random.uniform(5, 50)
            c = np.random.uniform(0, 180)
            v = np.random.uniform(5, 15)
        else:
            b, d, c, v = p0

        target = Object("Объект", b, d, c, v, self.observer, mode="bdcv")
        target.forward_movement(len(self.observer_coords[0]) - 1)
        self.target = target
        self.set_data()
        self.set_bearings()
        self.set_noise(seed=seed)

    def set_noise(self, seed=None):
        rng = np.random.RandomState(seed)
        self.bearings_with_noise = self.bearings.copy()
        noise = rng.normal(self.noise_mean, self.noise_std, len(self.bearings))
        self.bearings_with_noise += noise

    def set_data(self):
        time = np.arange(0, self.end_t + 1, self.tau)
        self.observer_data = np.vstack((self.observer_coords[:, time], time))
        self.target_data = np.vstack((np.array(self.target.coords)[:, time], time))
        self.true_params = np.array(self.target.get_params())

    def set_bearings(self):
        rx = self.target_data[0] - self.observer_data[0]
        ry = self.target_data[1] - self.observer_data[1]
        self.bearings = np.arctan2(ry, rx)
        self.distances = f.dist_func(self.observer_data, self.target_data)

    def get_observed_information(self):
        params = f.convert_to_xy(self.last_result)
        J = f.xy_func_jac(self.observer_data, params)
        I = J.T.dot(J) / (self.noise_std ** 2)
        return I

    def get_data(self):
        data = {
            "Время": self.observer_data[2],
            "Истинный пеленг": np.degrees(list(map(f.to_bearing, self.bearings))),
            "Расстояние": self.distances,
            "Зашумленный пеленг": np.degrees(
                list(map(f.to_bearing, self.bearings_with_noise))
            ),
        }
        return data