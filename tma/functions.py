import pandas as pd
import numpy as np


def bd_func(data, params):

    bearing, distance, course, velocity = params
    n = len(data[0])
    x_velocity = velocity * np.cos(course)
    y_velocity = velocity * np.sin(course)
    num = 1000 * distance * np.sin(bearing) + y_velocity * np.array(range(n)) - data[1]
    den = 1000 * distance * np.cos(bearing) + x_velocity * np.array(range(n)) - data[0]
    angle = np.arctan2(num, den)

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

    x_origin, y_origin, x_velocity, y_velocity = params
    r_y = 1000 * y_origin + y_velocity * data[2] - data[1]
    r_x = 1000 * x_origin + x_velocity * data[2] - data[0]
    angle = np.arctan2(r_y, r_x)

    return angle


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
    distance = np.linalg.norm(coords, 2, axis=0)
    angle = np.arctan2(y, x)

    return [angle, distance]


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


def get_df(res):
    try:
        algorithm_name = list(res[0].keys())[0]
    except AttributeError:
        res = np.ravel(res)
        algorithm_name = "ММП в реальном времени"
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