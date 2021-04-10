import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tma.functions as f


def plot_trajectories2(model):

    plt.plot(model.observer_data[0], model.observer_data[1])
    plt.plot(model.target_data[0], model.target_data[1])
    m = len(model.observer_data[0]) // 2
    plt.arrow(
        model.observer_coords[0][m - 30],
        model.observer_coords[1][m - 30],
        model.observer_coords[0][m + 1] - model.observer_coords[0][m],
        model.observer_coords[1][m + 1] - model.observer_coords[1][m],
        shape="full",
        lw=0,
        head_width=300,
        head_starts_at_zero=True,
    )
    plt.arrow(
        model.target_data[0][m - 30],
        model.target_data[1][m - 30],
        model.target_data[0][m + 1] - model.target_data[0][m],
        model.target_data[1][m + 1] - model.target_data[1][m],
        shape="full",
        lw=0,
        head_width=300,
        head_starts_at_zero=True,
        color="#ff7f0e",
    )
    plt.axis("square")
    plt.xlim(-5000, 10000)
    plt.ylim(-1000, 35000)
    plt.grid()
    ax = plt.gca()
    # if model.last_result is not None:
    #     from matplotlib.patches import Ellipse
    #     [b, d] = model.last_result[:2]
    #     [x, y] = [1000*d*np.cos(b), 1000*d*np.sin(b)]
    #     cov = model.last_cov
    #     eig = np.linalg.eig(cov)
    #     angle = np.arctan2(max(eig)/min(eig))
    #     confid = Ellipse([x, y], 1000, 1000)
    #     ax.add_artist(confid)
    ax.legend(["Наблюдатель", "Объект"])
    plt.show()


def plot_trajectories(model):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import pandas as pd

    x = model.observer_data[0][::8]
    y = model.observer_data[1][::8]
    df = pd.DataFrame({"x": x, "y": y})

    # plot setup
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(df["x"].min() - 100, df["x"].max() + 100)
    ax.set_ylim(df["y"].min() - 100, df["y"].max() + 100)
    xdata, ydata = [], []
    (ln,) = plt.plot([], [], "b--", alpha=0.5)

    def update(i):
        ln.set_data(df.iloc[0:i, 0], df.iloc[0:i, 1])
        return (ln,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(df),
        interval=100,
        blit=True,
    )
    ani.save("trajectory.gif")
    plt.show()


def plot_bearings(model, result):

    params = f.convert_to_xy(result["Метод N пеленгов"]["Полученные параметры"])
    sns.set_style("darkgrid")
    x = model.observer_data[2]
    y = np.degrees([f.to_bearing(i) for i in f.xy_func(model.observer_data, params)])
    sns.lineplot(
        x,
        np.degrees([f.to_bearing(i) for i in model.bearings]),
        linewidth=4,
    )

    sns.lineplot(x, y)

    ci = np.degrees(model.noise_std)
    ax = plt.gca()
    ax.fill_between(x, (y - ci), (y + ci), color="b", alpha=0.1)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Пеленг, °")
    ax.legend(["Истинный пеленг", "Расчетный пеленг"])
    plt.show()


def plot_trajectory(object):
    plt.plot(object.coords[0], object.coords[1], "b--", alpha=0.5)
    plt.axis("square")
    plt.show()
