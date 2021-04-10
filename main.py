import numpy as np
from tma.object import Object
from tma.tests import Tests
from tma.model import Model

# Пример моделирования

observer_x, observer_y, observer_course, observer_velocity = 0.0, 0.0, 0.0, 5.0
observer = Object(
    "Наблюдатель",
    observer_x,
    observer_y,
    observer_course,
    observer_velocity,
    verbose=True,
)

target_bearing, target_distance, target_course, target_velocity = (
    5.0,
    20.0,
    45.0,
    10.0,
)

target = Object(
    "Объект",
    target_bearing,
    target_distance,
    target_course,
    target_velocity,
    observer,
    mode="bdcv",
    verbose=True,
)

observer.forward_movement(3 * 60)
observer.change_course(270, "left", omega=0.5)
observer.forward_movement(5 * 60)
observer.change_course(90, "right", omega=0.5)
observer.forward_movement(3 * 60)

target.forward_movement(len(observer.coords[0]) - 1)

from tma.algorithms import mle_algorithm_v2, swarm
from tma.functions import get_df
from tma.plot import plot_trajectory, plot_bearings, plot_trajectories

model = Model(observer, target=target, noise_std=np.radians(1))
result = mle_algorithm_v2(model, p0=[1, 1, 1, 1])

# plot_bearings(model, result)

# target.forward_movement(7 * 60)
# target.change_course(270, "left", omega=0.5)
# target.forward_movement(len(observer.coords[0]) - len(target.coords[0]))

# dict_results = swarm(n=100, fixed_target=False, fixed_noise=False, p0=[0., 20., 45., 10.])
# df = f.get_df(dict_results)
