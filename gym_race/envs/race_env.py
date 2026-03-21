import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D

# =============================================================================
#  RaceEnv  –  Original wrapper (Pyrace-v1)
#  Discrete observation space (integers 0-10), 3 actions
# =============================================================================

class RaceEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode="human"):
        print("init RaceEnv (v1)")
        self.action_space      = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            np.array([0, 0, 0, 0, 0]),
            np.array([10, 10, 10, 10, 10]),
            dtype=int
        )
        self.is_view     = True
        self.pyrace      = PyRace2D(self.is_view)
        self.memory      = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs    = []
        self.pyrace  = PyRace2D(self.is_view, mode=self.render_mode)
        obs = self.pyrace.observe()
        return np.array(obs), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return np.array(obs), reward, done, False, {
            'dist':  self.pyrace.car.distance,
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


# =============================================================================
#  RaceEnvV2  –  Improved wrapper (Pyrace-v2)
#
#  Changes vs RaceEnv (v1):
#    - observation_space: continuous floats [0.0, 10.0] instead of integers
#    - action_space:      4 actions (added BRAKE)
# =============================================================================

class RaceEnvV2(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode="human"):
        print("init RaceEnvV2 (v2)")

        # 4 actions: 0=accelerate, 1=turn left, 2=turn right, 3=brake
        self.action_space = spaces.Discrete(4)

        # continuous radar readings in [0.0, 10.0]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([10.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32
        )

        self.is_view     = True
        self.pyrace      = PyRace2D(self.is_view)
        self.memory      = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs    = []
        self.pyrace  = PyRace2D(self.is_view, mode=self.render_mode)
        obs = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return np.array(obs, dtype=np.float32), reward, done, False, {
            'dist':  self.pyrace.car.distance,
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