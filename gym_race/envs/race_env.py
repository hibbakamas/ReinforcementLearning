import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D, PyRace2DV2

class RaceEnv(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}

    def __init__(self, render_mode="human", ):
        print("init")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            np.array([0, 0, 0, 0, 0]),
            np.array([10, 10, 10, 10, 10]),
            dtype=int
        )
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view)
        self.memory = []
        self.render_mode = render_mode
        self.msgs = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2D(self.is_view, mode=mode)
        obs = self.pyrace.observe()
        return np.array(obs), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return np.array(obs), reward, done, False, {
            'dist': self.pyrace.car.distance,
            'check': self.pyrace.car.current_check,
            'crash': not self.pyrace.car.is_alive
        }

    def render(self):
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class RaceEnvV2(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}

    def __init__(self, render_mode="human", ):
        print("init v2")
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )
        self.is_view = True
        self.pyrace = PyRace2DV2(self.is_view)
        self.memory = []
        self.render_mode = render_mode
        self.msgs = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2DV2(self.is_view, mode=mode)
        self.pyrace.car.radars.clear()
        for d in range(-90, 120, 45):
            self.pyrace.car.check_radar(d)
        obs = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), reward, done, False, {
            'dist': self.pyrace.car.distance,
            'check': self.pyrace.car.current_check,
            'crash': not self.pyrace.car.is_alive
        }

    def render(self):
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))