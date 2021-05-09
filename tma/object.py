import numpy as np
from tma.helper_functions import (
    to_angle,
    to_bearing,
    convert_to_xy,
    convert_to_bdcv,
)


class Object:
    def __init__(self, name, *args, mode="xycv", verbose=False):

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

        elif mode == "xy":
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
        self.name = name
        self.current_position = np.array(
            [self.x_origin, self.y_origin], dtype=np.float64
        )
        self.velocity = np.array(
            [self.x_velocity, self.y_velocity], dtype=np.float64
        )
        self.coords = [[self.x_origin], [self.y_origin]]

        if verbose:
            print(
                f"{self.name} имеет начальные "
                + f"параметры {convert_to_bdcv(self.true_params)}"
            )

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
                + f"{np.degrees(self.get_course()):.1f}°" 
                + f"(угловая скорость {omega}) за {t}с"
            )
