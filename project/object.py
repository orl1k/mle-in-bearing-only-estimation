import numpy as np
import matplotlib.pyplot as plt
import project.functions as f


class Object:
    def __init__(self, name, *args, mode="xycv", verbose=False):

        if mode == "xycv":
            self.x_origin, self.y_origin, observer_course, observer_velocity = args
            course = f.to_angle(np.radians(observer_course))
            self.x_velocity = observer_velocity * np.cos(course)
            self.y_velocity = observer_velocity * np.sin(course)

        elif mode == "bdcv":
            observer = args[4]
            (
                self.x_origin,
                self.y_origin,
                self.x_velocity,
                self.y_velocity,
            ) = f.convert_to_xy(args[:4])
            self.x_origin *= 1000
            self.y_origin *= 1000

        elif mode == "xy":
            self.x_origin, self.y_origin, self.x_velocity, self.y_velocity = args

        # fix true_params when xy
        self.verbose = verbose
        self.true_params = list(args[:4])
        self.name = name
        self.position = np.array([self.x_origin, self.y_origin], dtype=np.float64)
        self.velocity = np.array([self.x_velocity, self.y_velocity], dtype=np.float64)
        self.coords = [[self.x_origin], [self.y_origin]]
        self.vels = [[self.x_velocity], [self.y_velocity]]

    def get_course(self):
        angle = np.arctan2(self.velocity[1], self.velocity[0])
        return f.to_bearing(angle)

    def get_params(self):
        return self.true_params.copy()

    def forward_movement(self, time):
        for i in range(time):
            self.update_position()
        if self.verbose:
            print(
                "{} движется прямо по курсу {:.1f}° {}с".format(
                    self.name, np.degrees(self.get_course()), time
                )
            )

    def update_position(self):
        self.position += self.velocity
        self.coords[0].append(self.position[0])
        self.coords[1].append(self.position[1])
        self.vels[0].append(self.velocity[0])
        self.vels[1].append(self.velocity[1])

    def update_velocity(self, theta):
        """ поворот вектора скорости на угол theta """
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        self.velocity = self.velocity.dot(R)

    def change_course(self, new_course, turn, radius=None, omega=0.5, stop_time=0):
        """ поворот на новый курс new_course """
        new_course = np.radians(new_course)
        if radius != None:
            theta = np.pi - 2 * np.arccos(
                np.linalg.norm((self.velocity)) / (2 * radius)
            )
        else:
            theta = np.radians(omega)

        if turn == "left":
            theta = -theta
        t = 0
        if stop_time != 0:
            while t < stop_time:
                t += 1
                self.update_velocity(theta)
                self.update_position()
        else:
            while abs(self.get_course() - new_course) % 2 * np.pi > abs(theta):
                t += 1
                self.update_velocity(theta)
                self.update_position()
        if self.verbose:
            print(
                "{} перешёл на курс {:.1f}° за {}с".format(
                    self.name, np.degrees(self.get_course()), t
                )
            )
