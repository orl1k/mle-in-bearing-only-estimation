import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
    Класс предназначен для сохранения результатов.

    Форматы:
    Пеленг - .x градусов
    Дистанция - .xxx м
    Курс - .x градусов
    Скорость - .xx м / с

    save_results() предназначен для сохранения результатов метода tma.swarm()
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

    def get_df(self, res):
        algorithm_name = list(res[0].keys())[0]
        df = pd.DataFrame(
            columns=[
                "П0_ист",
                "Д0_ист",
                "К0_ист",
                "V0_ист",
                "П0_расч",
                "Д0_расч",
                "К0_расч",
                "V0_расч",
                "П0_апр",
                "Д0_апр",
                "К0_апр",
                "V0_апр",
                "Птек_ист",
                "Дтек_ист",
                "Птек_расч",
                "Дтек_расч",
                "СКО X",
                "СКО Y",
                "СКО VX",
                "СКО VY",
                "Ка",
                "Кб",
                "Успех",
                "t",
                "Nf",
                "Iter",
            ]
        )
        for i, r in enumerate(res):
            r = r[algorithm_name]
            flat_list = [item for sublist in r.values() for item in sublist]
            df.loc[i] = flat_list
        return df

    def save_df(self, df, name="tests/results.xlsx"):
        df.to_excel(name, index=False)

    def save_bearings_fig(self, true_distances, pred_distances):
        plt.plot(true_distances, linewidth=5.0)
        plt.plot(2)
        ax = plt.gca()
        ax.set_xlabel("время (с)")
        ax.set_ylabel("дистанция (м)")
        ax.legend(["Истинное расстояние", "Расчетное расстояние"])
        plt.xlim([0, len(true_distances) - 1])
        plt.savefig("Пеленги" + self.test_name + ".png")
