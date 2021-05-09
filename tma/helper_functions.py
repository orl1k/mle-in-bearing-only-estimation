import pandas as pd
import numpy as np
from collections import namedtuple


def bd_func(data, params):

    b, d, c, v = params
    n = len(data[0])
    vx = v * np.cos(c)
    vy = v * np.sin(c)
    r_y = 1000 * d * np.sin(b) + vy * np.array(range(n)) - data[1]
    r_x = 1000 * d * np.cos(b) + vx * np.array(range(n)) - data[0]
    angle = np.arctan2(r_y, r_x)

    return angle


def bd_func_jac(data, params):

    J = []
    b, d, c, v = params
    n = len(data[0])

    vx = v * np.cos(c)
    vy = v * np.sin(c)
    x = d * np.cos(b)
    y = d * np.sin(b)

    r_y = 1000 * y + vy * np.arange(n) - data[1]
    r_x = 1000 * x + vx * np.arange(n) - data[0]
    R2 = r_x ** 2 + r_y ** 2

    J.append((1000 * (x * r_x + y * r_y)) / R2)
    J.append((1000 * (np.sin(b) * r_x - np.cos(b) * x * r_y)) / R2)
    J.append((vx * r_x + vy * r_y) * np.arange(n) / R2)
    J.append((np.sin(c) * r_x - np.cos(c) * r_y) * np.arange(n) / R2)

    return np.array(J).T


def xy_func(data, params):

    x, y, vx, vy = params
    r_y = 1000 * y + vy * data[2] - data[1]
    r_x = 1000 * x + vx * data[2] - data[0]
    b = np.arctan2(r_y, r_x)

    return b


def xy_func_jac(data, params):

    J = []
    x, y, vx, vy = params
    r_y = 1000 * y + vy * data[2] - data[1]
    r_x = 1000 * x + vx * data[2] - data[0]
    R2 = r_x ** 2 + r_y ** 2

    J.append(-1000 * r_y / R2)
    J.append(1000 * r_x / R2)
    J.append(-(data[2] * r_y) / R2)
    J.append((data[2] * r_x) / R2)

    return np.array(J).T


def dist_func(r_obs, r_tar):
    return np.linalg.norm(r_obs - r_tar, 2, axis=0) / 1000


def get_random_p0(seed=None):

    rng = np.random.RandomState(seed)
    b = 0
    d = rng.uniform(5, 50)
    c = rng.uniform(0, 360)
    v = rng.uniform(5, 25)

    return [b, d, c, v]


def to_bearing(a):

    """ угол в радианах -> пеленг в радианах """
    a = (np.pi / 2) - ((2 * np.pi + a) * (a < 0) + a * (a >= 0))

    return a if a >= 0 else a + 2 * np.pi


def to_angle(b):

    """ пеленг в радианах -> угол в радианах """
    angle = (np.pi / 2) - b

    return angle if abs(angle) <= np.pi else (2 * np.pi) + angle


def convert_to_polar(coords):

    x, y = coords
    a = np.linalg.norm(coords, 2, axis=0)
    b = np.arctan2(y, x)

    return [b, a]


def convert_to_bdcv(params):

    x, y, vx, vy = params
    b, d = convert_to_polar([x, y])
    c, v = convert_to_polar([vx, vy])
    b = np.degrees(to_bearing(b))
    c = np.degrees(to_bearing(c))

    return np.array([b, d, c, v])


def convert_to_xy(params):

    b, d, c, v = params
    b = to_angle(np.radians(b))
    c = to_angle(np.radians(c))
    x = d * np.cos(b)
    y = d * np.sin(b)
    vx = v * np.cos(c)
    vy = v * np.sin(c)

    return np.array([x, y, vx, vy])


def get_df(result):

    mapper = {
        "b0": "П0_ист",
        "d0": "Д0_ист",
        "c0": "К0_ист",
        "v0": "V0_ист",
        "res_b0": "П0_расч",
        "res_d0": "Д0_расч",
        "res_c0": "К0_расч",
        "res_v0": "V0_расч",
        "init_b0": "П0_апр",
        "init_d0": "Д0_апр",
        "init_c0": "К0_апр",
        "init_v0": "V0_апр",
        "cur_b0": "Птек_ист",
        "cur_d0": "Дтек_ист",
        "cur_res_b0": "Птек_расч",
        "cur_res_d0": "Дтек_расч",
        "std_x": "СКО X",
        "std_y": "СКО Y",
        "std_vx": "СКО VX",
        "std_vy": "СКО VY",
        "ka": "Ка",
        "kb": "Кб",
        "kc": "Успех",
        "t": "Время",
        "nf": "Вычисления",
        "iter": "Итерации",
    }

    if isinstance(result, list):
        result_list = map(parser, result)
    else:
        result_list = map(parser, [result])

    return pd.DataFrame(result_list).rename(columns=mapper)


def parser(result):
    parsed_res = namedtuple(
        "res",
        [
            "b0",
            "d0",
            "c0",
            "v0",
            "res_b0",
            "res_d0",
            "res_c0",
            "res_v0",
            "init_b0",
            "init_d0",
            "init_c0",
            "init_v0",
            "cur_b0",
            "cur_d0",
            "cur_res_b0",
            "cur_res_d0",
            "std_x",
            "std_y",
            "std_vx",
            "std_vy",
            "ka",
            "kb",
            "kc",
            "t",
            "nf",
            "iter",
        ],
    )

    return parsed_res(*(i for sublist in result for i in sublist))


def df_to_docx(df, path):

    import docx

    doc = docx.Document()
    t = doc.add_table(df.shape[0] + 2, df.shape[1])
    t.style = "Table Grid"

    for j in range(df.shape[-1]):
        t.cell(0, j).text = df.columns[j]

    for i in range(df.shape[0]):
        for j in range(df.shape[-1]):
            t.cell(i + 1, j).text = str(df.values[i, j])

    doc.save(path)