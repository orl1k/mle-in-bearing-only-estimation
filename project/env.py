import gym
from gym import spaces
from project.botma import TMA
import numpy as np
from project.ship import Ship

class CustomGym(gym.Env):
    def __init__(self, observer):
        self.observer = observer
        S0 = TMA.get_random_p0()
        self.target = Ship('Объект', S0[0], S0[1], S0[2], S0[3], self.observer, mode='bdcv')
        self.observation = 0
        self.low = np.array([-0.5])
        self.high = np.array([0.5])
        self.action_space = spaces.Box(self.low, self.high)
        self.observation_space = spaces.Box(np.array([-2*np.pi, -2*np.pi]), np.array([2*np.pi, 2*np.pi]))
        self.t = 0
        self.sum_reward = 0

    def step(self, action):
        self.observer.change_course(0, 'right', omega=action[0], stop_time=180)
        self.t += 180
        self.target.forward_movement(180)

        self.observation += 1
        if self.observation < 10:
            done = False
        else:
            done = True

        self.tma = TMA(self.observer, self.target, sd=np.radians(0.1), seed=1)
        self.tma.mle_algorithm_v6([1, 1, 1, 1])
        sr = np.sum(np.diag(self.tma.get_observed_information()))
        if self.tma.last_full_result['ММП v6']['Оценка'][1] > 10:
            sr = 0
        reward = sr - self.sum_reward
        self.sum_reward += reward
        # reward -= self.tma.distance_array[-1] * 1e10
        reward = -self.tma.distance_array[-1]
        observation = self.tma.bearings_with_noise[-1], Ship.to_angle(self.observer.get_course())

        return observation, reward, done, {}
    
    def reset(self):
        self.observation = 0
        self.t = 0
        self.sum_reward = 0
        self.observer = Ship('Наблюдатель', 0, 0, 0, 3)
        S0 = TMA.get_random_p0()
        self.target = Ship('Объект', S0[0], S0[1], S0[2], S0[3], self.observer, mode='bdcv')
        observation = Ship.to_angle(np.radians(S0[0])), Ship.to_angle(self.observer.get_course())
        return observation  # reward, done, info can't be included
