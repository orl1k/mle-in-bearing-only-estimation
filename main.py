import numpy as np
from project.object import Object
from project.tests import Tests
from project.botma import TMA

# Класс для сохранения результатов
tests = Tests("1")

# Пример моделирования

# Создаем наблюдатель
observer_x, observer_y, observer_course, observer_velocity = 0.0, 0.0, 0.0, 3.0
observer = Object(
    "Наблюдатель",
    observer_x,
    observer_y,
    observer_course,
    observer_velocity,
    verbose=True,
)
# Создаем объект
target_bearing, target_distance, target_course, target_velocity = (
    0.0,
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

# Моделирование траекторий
observer.forward_movement(3 * 60)
observer.change_course(270, "left", omega=0.5)
observer.forward_movement(5 * 60)
observer.change_course(90, "right", omega=0.5)
observer.forward_movement(3 * 60)

# Время движения объекта должно совпадать с временем наблюдателя для TMA
target.forward_movement(len(observer.coords[0]) - 1)

# # Рассматривается маневр объекта
# target.forward_movement(7 * 60)
# target.change_course(270, "left", omega=0.5)
# target.forward_movement(len(observer.coords[0]) - len(target.coords[0]))

# # Запуск множества моделей
# dict_results = tma.swarm(n=100, fixed_target=False, fixed_noise=False, p0=[0., 20., 45., 10.])
# df = tests.get_df(dict_results)
# tests.save_df(df)
