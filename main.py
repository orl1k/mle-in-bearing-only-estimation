import numpy as np
from tma.object import Observer, Target
from tma.model import Model
from tma.algorithms import Algorithms
from tma.helper_functions import get_df, convert_to_xy

observer_x, observer_y, observer_course, observer_velocity = 0.0, 0.0, 0.0, 5.0
observer = Observer(
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

target = Target(
    observer,
    target_bearing,
    target_distance,
    target_course,
    target_velocity,
    verbose=True,
)

observer.forward_movement(3 * 60)
observer.change_course(270, "left", omega=0.5)
observer.forward_movement(5 * 60)
observer.change_course(90, "right", omega=0.5)
observer.forward_movement(3 * 60)

target.forward_movement(len(observer.coords[0]) - 1)

model = Model(observer, target=target, verbose=True, seed=1)
alg = Algorithms(model)

p0 = convert_to_xy([0.0, 25.0, 90.0, 7.0])
res = alg.mle_v2(p0)
alg.print_result(res)

print(observer)
print(target)