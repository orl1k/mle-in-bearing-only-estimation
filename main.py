import numpy as np
from project.ship import Ship
from project.tests import Tests
from project.botma import TMA

# Класс для сохранения результатов
tests = Tests("2")

# Пример моделирования

# Создаем наблюдатель
observer_x, observer_y, observer_course, observer_velocity = 0, 0, 0, 3
observer = Ship(
    "Наблюдатель",
    observer_x,
    observer_y,
    observer_course,
    observer_velocity,
    verbose=True,
)
# Создаем объект
target_bearing, target_distance, target_course, target_velocity = 0, 20, 45, 10
target = Ship(
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
# target.forward_movement(10 * 60)
# target.change_course(270, 'left', omega=0.5)
# target.forward_movement(360)

# tma = TMA(observer, target=target)
# print(tma.mle_algorithm_v2([1, 1, 1, 1]))
# tma.plot_trajectories()

# tma = TMA(observer, std=np.radians(0.1))
# d_r = tma.swarm(n=10, algorithm_name="Метод N пеленгов", p0=[0, 20, 45, 10])
# df = tests.get_df(d_r)
# print(df.drop(["П0_ист", "Д0_ист", "К0_ист"], axis=1).round(2))

# # Запуск множества моделей
# dict_results = tma.swarm(n=100, fixed_target=False, fixed_noise=False, p0=[0., 20., 45., 10.])
# df = tests.get_df(dict_results)
# tests.save_df(df)
