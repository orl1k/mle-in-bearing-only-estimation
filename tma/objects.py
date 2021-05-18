import numpy as np
from abc import ABC, abstractmethod
from tma.helper_functions import (
    to_angle,
    to_bearing,
    convert_to_xy,
    convert_to_bdcv,
)


class BaseObject(ABC):
    @abstractmethod
    def __init__(self, *args, mode="xycv", verbose=False):

        if mode == "xycv":
            self.x_origin, self.y_origin, course, velocity = args
            course = to_angle(np.radians(course))
            self.x_velocity = velocity * np.cos(course)
            self.y_velocity = velocity * np.sin(course)

        elif mode == "bdcv":
            observer = args[4]
            (
                self.x_origin,
                self.y_origin,
                self.x_velocity,
                self.y_velocity,
            ) = convert_to_xy(args[:4])
            self.x_origin += observer.x_origin
            self.y_origin += observer.y_origin

        elif mode == "xyv":
            (
                self.x_origin,
                self.y_origin,
                self.x_velocity,
                self.y_velocity,
            ) = args

        self.true_params = (
            self.x_origin,
            self.y_origin,
            self.x_velocity,
            self.y_velocity,
        )

        self.x_origin *= 1000
        self.y_origin *= 1000
        self.verbose = verbose
        self.current_position = np.array(
            [self.x_origin, self.y_origin], dtype=np.float64
        )
        self.velocity = np.array(
            [self.x_velocity, self.y_velocity], dtype=np.float64
        )
        self.coords = [[self.x_origin], [self.y_origin]]

    def get_course(self):
        angle = np.arctan2(*self.velocity[::-1])
        return to_bearing(angle)

    def get_params(self):
        return self.true_params

    def forward_movement(self, time):
        for _ in range(time):
            self.update_current_position()
        if self.verbose:
            print(
                "{} движется прямо по курсу {:.1f}° {}с".format(
                    self.name, np.degrees(self.get_course()), time
                )
            )

    def update_current_position(self):
        self.current_position += self.velocity
        self.coords[0].append(self.current_position[0])
        self.coords[1].append(self.current_position[1])

    def update_velocity(self, theta):
        """ поворот вектора скорости на угол theta """

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        self.velocity = self.velocity.dot(R)

    def change_course(self, new_course, turn, radius=None, omega=0.5):
        """ поворот на новый курс new_course """

        if radius != None:
            theta = np.pi - 2 * np.arccos(
                np.linalg.norm(self.velocity) / (2 * radius)
            )
        else:
            theta = np.radians(omega)

        if turn == "left":
            theta = -theta

        new_course = np.radians(new_course)
        threshold = abs(theta) - 1e-10
        t = 0

        while abs(new_course - self.get_course()) % (2 * np.pi) > threshold:
            t += 1
            self.update_velocity(theta)
            self.update_current_position()

        if self.verbose:
            print(
                f"{self.name} перешёл на курс "
                + f"{np.degrees(self.get_course()):.1f}° "
                + f"(угловая скорость {omega}) за {t}с"
            )

    @staticmethod
    def print_initial(name, args):
        print(f"{name} имеет начальные " + f"параметры {args}")

    def __repr__(self):
        return self.__class__.__name__ + str(self.params)


class Observer(BaseObject):
    def __init__(self, *args, verbose=False, mode="xycv"):
        super().__init__(*args, mode=mode, verbose=verbose)
        self.params = args
        self.name = "Наблюдатель"
        if verbose:
            super().print_initial(self.name, args)


class Target(BaseObject):
    def __init__(self, observer, *args, verbose=False, mode="bdcv"):
        super().__init__(*args, observer, mode=mode, verbose=verbose)
        self.params = args
        self.name = "Объект"
        if verbose:
            super().print_initial(self.name, args)
