import matplotlib.pyplot as plt


def plot_trajectories(self):

    plt.plot(self.observer_data[0], self.observer_data[1])
    plt.plot(self.target_data[0], self.target_data[1])
    m = len(self.observer_data[0]) // 2
    plt.arrow(
        self.observer_coords[0][m - 30],
        self.observer_coords[1][m - 30],
        self.observer_coords[0][m + 1] - self.observer_coords[0][m],
        self.observer_coords[1][m + 1] - self.observer_coords[1][m],
        shape="full",
        lw=0,
        head_width=300,
        head_starts_at_zero=True,
    )
    plt.arrow(
        self.target_data[0][m - 30],
        self.target_data[1][m - 30],
        self.target_data[0][m + 1] - self.target_data[0][m],
        self.target_data[1][m + 1] - self.target_data[1][m],
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
    # if self.last_result is not None:
    #     from matplotlib.patches import Ellipse
    #     [b, d] = self.last_result[:2]
    #     [x, y] = [1000*d*np.cos(b), 1000*d*np.sin(b)]
    #     cov = self.last_cov
    #     eig = np.linalg.eig(cov)
    #     angle = np.arctan2(max(eig)/min(eig))
    #     confid = Ellipse([x, y], 1000, 1000)
    #     ax.add_artist(confid)
    ax.legend(["Наблюдатель", "Объект"])
    plt.show()


def plot_bearings(self):

    plt.plot(
        self.observer_data[2],
        np.degrees([Object.to_bearing(i) for i in self.bearings]),
        linewidth=5.0,
    )
    plt.plot(
        self.observer_data[2],
        np.degrees(
            [
                Object.to_bearing(i)
                for i in f.xy_func(
                    self.observer_data, Object.convert_to_xy(self.last_result)
                )
            ]
        ),
    )
    ax = plt.gca()
    ax.set_xlabel("время (с)")
    ax.set_ylabel("пеленг (г)")
    ax.legend(["Истинный пеленг", "Расчетный пеленг"])
    plt.show()


def plot_trajectory(self):
    plt.plot(self.coords[0], self.coords[1])
    plt.axis("square")
    plt.show()
