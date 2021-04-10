import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
    Класс предназначен для сохранения результатов.

    Форматы:
    Пеленг - .x градусов
    Дистанция - .xxx м
    Курс - .x градусов
    Скорость - .xx м / с

"""


class Tests:
    def __init__(self, name):
        self.test_name = name
        self.results = {}

    def save_bearing_array(self, bearing_array):
        with open("bearings.txt", "ab") as f:
            np.savetxt(f, np.degrees(bearing_array), fmt="%.1f", newline=" ")
            f.write(b"\n")

    def save_distance_array(self, distance_array):
        with open("distances.txt", "ab") as f:
            np.savetxt(f, distance_array / 1000, fmt="%.3f", newline=" ")
            f.write(b"\n")

    def save_bearings_fig(self, true_distances, pred_distances):
        plt.plot(true_distances, linewidth=5.0)
        plt.plot(2)
        ax = plt.gca()
        ax.set_xlabel("время (с)")
        ax.set_ylabel("дистанция (м)")
        ax.legend(["Истинное расстояние", "Расчетное расстояние"])
        plt.xlim([0, len(true_distances) - 1])
        plt.savefig("Пеленги" + self.test_name + ".png")
