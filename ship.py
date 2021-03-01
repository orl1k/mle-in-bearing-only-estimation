import numpy as np
import matplotlib.pyplot as plt

class Ship():

    @staticmethod
    def transform_to_bearing(angle):
        """ угол -> пеленг в радианах """
        angle = (np.pi / 2) - ((2 * np.pi + angle) *
                               (angle < 0) + angle * (angle >= 0))
        return angle if angle >= 0 else angle + 2 * np.pi

    @staticmethod
    def transform_to_angle(bearing):
        """ пеленг -> угол в радианах """
        angle = (np.pi / 2) - bearing
        return angle if abs(angle) <= np.pi else (2 * np.pi) + angle

    @staticmethod
    def convert_to_polar(x, y):
        distance = (x**2 + y**2)**(0.5)
        angle = np.arctan2(y, x)
        return [angle, distance]

    @staticmethod
    def convert_to_xy(params):
        bearing, distance, course, velocity = params
        bearing = Ship.transform_to_angle(np.radians(bearing))
        x_coord = 1000 * distance * np.cos(bearing)
        y_coord = 1000 * distance * np.sin(bearing)
        course = Ship.transform_to_angle(np.radians(course))
        x_velocity = velocity * np.cos(course)
        y_velocity = velocity * np.sin(course)
        return [x_coord, y_coord, x_velocity, y_velocity]

    def __init__(self, name, *args, mode='xycv'):

        if mode == 'xycv':
            self.x_origin, self.y_origin, observer_course, observer_velocity = args
            course = self.transform_to_angle(np.radians(observer_course))
            self.x_velocity = observer_velocity * np.cos(course)
            self.y_velocity = observer_velocity * np.sin(course)

        if mode == 'bdcv':
            observer = args[4]
            self.x_origin, self.y_origin, self.x_velocity, self.y_velocity = self.convert_to_xy(
                args[:4])

        elif mode == 'xy':
            self.x_origin, self.y_origin, self.x_velocity, self.y_velocity = args

        # fix true_params when xy
        self.true_params = list(args[:4])
        self.name = name
        self.position = np.array((self.x_origin, self.y_origin))
        self.velocity = np.array((self.x_velocity, self.y_velocity))
        self.coords = [[self.x_origin], [self.y_origin]]
        self.vels = [[self.x_velocity], [self.y_velocity]]
        
    def plot_trajectory(self):
        plt.plot(self.coords[0], self.coords[1])
        plt.axis('square')
        plt.show()

    def print_current_position(self, mode='not default', lang='ru'):
        with np.printoptions(precision=3, suppress=True,
                             formatter={'float': '{: 0.3f}'.format}):
            if mode == 'default':
                if lang == 'eng':
                    print('coordinates: {}, velocity: {}'.format(
                          self.position, self.velocity))
                elif lang == 'ru':
                    print('coordinates: {}, скорость: {}м/с'.format(
                          self.position, self.velocity))
            else:
                if lang == 'eng':
                    print('coords: {}, velocity: {}, course: {}'.format(
                          self.position, np.linalg.norm(self.velocity),
                          np.degrees(self.get_course())))
                if lang == 'ru':
                    print('координаты: {}, скорость: {:.3f}м/с, курс: {:.3f}°'.format(
                          self.position, np.linalg.norm(self.velocity),
                          np.degrees(self.get_course())))

    def get_course(self):
        angle = np.arctan2(self.velocity[1], self.velocity[0])
        return self.transform_to_bearing(angle)

    def get_params(self):
        return self.true_params

    def forward_movement(self, time):
        for i in range(time):
            self.update_position()
        print('{} движется прямо по курсу {:.1f}° {}с'.format(self.name,
                                                                      np.degrees(self.get_course()), time))

    def update_position(self):
        self.position = self.position + self.velocity
        self.coords[0].append(self.position[0])
        self.coords[1].append(self.position[1])
        self.vels[0].append(self.velocity[0])
        self.vels[1].append(self.velocity[1])

    def update_velocity(self, teta):
        """ rotate velocity vector by teta """
        c, s = np.cos(teta), np.sin(teta)
        R = np.array(((c, -s), (s, c)))
        self.velocity = self.velocity.dot(R)

    def change_course(self, new_course, turn, radius=None, omega=0.5):
        """ change ship's course from current to new_course """
        new_course = np.radians(new_course)
        # teta = (np.arctan2(self.velocity[1],
        #                    self.velocity[0]) - new_course) / time
        if radius != None:
            teta = np.pi - 2 * \
                np.arccos(np.linalg.norm((self.velocity)) / (2 * radius))
        else:
            omega = np.radians(omega)
            teta = omega

        eps = teta
        if turn == 'left':
            teta = - teta
        time = 0
        while abs(self.get_course() - new_course) % 2 * np.pi > eps:
            time += 1
            self.update_velocity(teta)
            self.update_position()
        print('{} перешёл на курс {:.1f}° за {}с'.format(self.name,
                                                                 np.degrees(self.get_course()), time))
