import numpy as np
from tma.objects import Target
from tma.helper_functions import dist_func, to_bearing


class Model:
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
        if target is None:
            self.new_target()
        else:
            if len(target.coords[0]) != len(observer.coords[0]):
                raise ValueError(
                    "Время движения наблюдателя "
                    + "и объека должны быть одинаковыми"
                )
            self.target = target
            self.set_data()
            self.set_bearings()
        self.set_noise(seed=seed)
        if verbose:
            self.verbose = verbose
            print(
                "Параметры модели: "
                + f"СКО = {np.degrees(self.noise_std):.1f}, "
                + f"tau = {self.tau}, "
                + f"end_time = {self.end_t}"
            )

    def new_target(self, p0=None):
        b, d, c, v = [5.0, 20.0, 45.0, 10.0] if p0 is None else p0
        target = Target(self.observer, b, d, c, v)
        target.forward_movement(len(self.observer_coords[0]) - 1)
        self.target = target
        self.set_data()
        self.set_bearings()

    def set_noise(self, seed=None):
        rng = np.random.RandomState(seed)
        self.bearings_with_noise = self.bearings.copy()
        noise = rng.normal(self.noise_mean, self.noise_std, len(self.bearings))
        self.bearings_with_noise += noise

    def set_data(self):
        time = np.arange(0, self.end_t + 1, self.tau)
        self.observer_data = np.vstack((self.observer_coords[:, time], time))
        self.target_data = np.vstack(
            (np.array(self.target.coords)[:, time], time)
        )
        self.true_params = np.array(self.target.get_params())

    def set_bearings(self):
        rx = self.target_data[0] - self.observer_data[0]
        ry = self.target_data[1] - self.observer_data[1]
        self.bearings = np.arctan2(ry, rx)
        self.distances = dist_func(self.observer_data, self.target_data)

    def get_data(self):
        data = {
            "Время": self.observer_data[2],
            "Истинный пеленг": np.degrees(
                list(map(to_bearing, self.bearings))
            ),
            "Расстояние": self.distances,
            "Зашумленный пеленг": np.degrees(
                list(map(to_bearing, self.bearings_with_noise))
            ),
        }
        return data
