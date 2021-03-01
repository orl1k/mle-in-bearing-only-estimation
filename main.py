import numpy as np
from ship import Ship
from tests import Tests
from botma import TMA
import lm
import time

tests = Tests('1')

# Пример моделирования

# Создаем наблюдатель
observer_x, observer_y, observer_course, observer_velocity = 0, 0, 0, 3
observer = Ship('Наблюдатель', observer_x, observer_y, observer_course,
                observer_velocity)
# Создаем объект

target_bearing, target_distance, target_course, target_velocity = 0, 20, 45, 10
target = Ship('Объект', target_bearing, target_distance, target_course,
              target_velocity, observer, mode='bdcv')
# Моделирование траекторий
observer.forward_movement(3 * 60)
observer.change_course(270, 'left', omega=0.5)
observer.forward_movement(5 * 60)
observer.change_course(90, 'right', omega=0.5)
observer.forward_movement(120)

# Время движения объекта должно совпадать с временем наблюдателя для TMA
target.forward_movement(len(observer.coords[0])-1)

# # Рассматривается маневр объекта
# target.forward_movement(10 * 60)
# target.change_course(270, 'left', omega=0.5)
# target.forward_movement(360)

tma = TMA(observer, target, sd = np.radians(0.3))

print(tma.mle_algorithm_v4([1, 1, 1, 1]))

# tma.n_bearings_algorithm(tma.bearings_with_noise[0])

# r = tma.swarm(100)
# tests.save_results(r)

# # Запуск множества моделей
# r = tma.swarm(100)
# tests.save_results(r)
