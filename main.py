import numpy as np
import pandas as pd
from ship import Ship
from tests import Tests
from botma import TMA

tests = Tests('03')

# Пример моделирования 

# Создаем корабль-наблюдатель
observer_x, observer_y, observer_course, observer_velocity = 0, 0, 0, 3
observer = Ship('наблюдатель', observer_x, observer_y, observer_course,
                observer_velocity)  
# Создаем корабль-объект
target_bearing, target_distance, target_course, target_velocity = 0, 20, 45, 10
target = Ship('объект', target_bearing, target_distance, target_course,
              target_velocity, observer, mode='bdcv')  
# Моделирование траекторий
observer.forward_movement(3 * 60)
observer.change_course(270, 'left', omega=0.5)
observer.forward_movement(5 * 60)
observer.change_course(90, 'right', omega=0.5)
observer.forward_movement(120)
# Время движения объекта должно совпадать с временем наблюдателя для TMA
target.forward_movement(len(observer.coords[0]))

# # Рассматривается маневр объекта
# target.forward_movement(10 * 60)
# target.change_course(270, 'left', omega=0.5)
# target.forward_movement(360)

np.random.seed(207)

tma = TMA(observer, target, standart_deviation=3 * np.pi / 180)
print(tma.mle_algorithm_v3([1, 1, 1, 1]))


# # Запуск множества моделей
# r = tma.swarm(100)
# tests.save_results(r)
