import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

"""

    Класс предназначен для сохранения результатов.

    Форматы:
    Пеленг - .x градусов
    Дистанция - .xxx м
    Курс - .x градусов
    Скорость - .xx м / с

    save_results() предназначен для сохранения результатов метода tma.swarm()
"""


class Tests():
    def __init__(self, name):
        self.test_name = name
        self.results = {}

    def save_bearing_array(self, bearing_array):
        with open('bearings.txt', 'ab') as f:
            np.savetxt(f, np.degrees(bearing_array),
                       fmt='%.1f', newline=' ')
            f.write(b"\n")

    def save_distance_array(self, distance_array):
        with open('distances.txt', 'ab') as f:
            np.savetxt(f, distance_array /
                       1000, fmt='%.3f', newline=' ')
            f.write(b"\n")

    def save_results(self, res):
        algorithm_name = list(res[0].keys())[0]
        for i, r in enumerate(res):
            R = []
            r = r[algorithm_name]
            del r['Данные']
            R.append(r['Истинные параметры'])
            R.append(r['Полученные параметры'])
            R.append(r['Начальное приближение'])
            R.append(r['Среднеквадратичное отклонение параметров'])
            R.append(r['Оценка'])
            R.append(r['Время работы'])
            R.append(r['Число вычислений функции'])
            R = [i for j in R for i in j]
            res[i] = R
        with xlsxwriter.Workbook('results.xlsx') as workbook:
            worksheet = workbook.add_worksheet()
            for row_num, data in enumerate(res):
                worksheet.write_row(row_num, 0, data)

    def save_bearings_fig(self, true_distances, pred_distances):
        plt.plot(true_distances, linewidth=5.0)
        plt.plot(2)
        ax = plt.gca()
        ax.set_xlabel('время (с)')
        ax.set_ylabel('дистанция (м)')
        ax.legend(['Истинное расстояние', 'Расчетное расстояние'])
        plt.xlim([0, len(true_distances) - 1])
        plt.savefig('Пеленги' + self.test_name + '.png')
