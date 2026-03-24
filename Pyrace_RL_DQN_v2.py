import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

import gymnasium as gym
import gym_race

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    BACKEND = 'torch'
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers
    BACKEND = 'keras'

VERSION_NAME = 'DQN_v02'

REPORT_EPISODES  = 500
DISPLAY_EPISODES = 100

DISCOUNT_FACTOR  = 0.99
LEARNING_RATE    = 1e-3
BATCH_SIZE       = 64
MEMORY_SIZE      = 50_000
MIN_MEMORY       = 1_000
MIN_EXPLORE_RATE = 0.01
MAX_EXPLORE_RATE = 1.0
DECAY_FACTOR     = 500
NUM_EPISODES     = 65_000
MAX_T            = 2_000


def build_model_torch(state_size, action_size):
    model = nn.Sequential(
        nn.Linear(state_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_size),
    )
    return model


def build_model_keras(state_size, action_size):
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='mse'
    )
    return model


class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size
        self.memory      = ReplayBuffer(MEMORY_SIZE)

        if BACKEND == 'torch':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = build_model_torch(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
            self.loss_fn = nn.MSELoss()
        else:
            self.model = build_model_keras(state_size, action_size)

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            return random.randrange(self.action_size)
        return self._predict_best_action(state)

    def _predict_best_action(self, state):
        s = np.array(state, dtype=np.float32)

        if BACKEND == 'torch':
            with torch.no_grad():
                t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
                q = self.model(t)
            return int(q.argmax().item())
        else:
            q = self.model.predict(s.reshape(1, -1), verbose=0)
            return int(np.argmax(q[0]))

    def learn(self):
        if len(self.memory) < MIN_MEMORY:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        if BACKEND == 'torch':
            self._learn_torch(states, actions, rewards, next_states, dones)
        else:
            self._learn_keras(states, actions, rewards, next_states, dones)

    def _learn_torch(self, states, actions, rewards, next_states, dones):
        S  = torch.FloatTensor(states).to(self.device)
        A  = torch.LongTensor(actions).to(self.device)
        R  = torch.FloatTensor(rewards).to(self.device)
        S2 = torch.FloatTensor(next_states).to(self.device)
        D  = torch.FloatTensor(dones).to(self.device)

        q_pred = self.model(S).gather(1, A.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_q = self.model(S2).max(1)[0]

        q_target = R + DISCOUNT_FACTOR * best_q * (1 - D)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _learn_keras(self, states, actions, rewards, next_states, dones):
        q_values = self.model.predict(states, verbose=0)
        q_next   = self.model.predict(next_states, verbose=0)
        best_q   = np.max(q_next, axis=1)
        targets  = rewards + DISCOUNT_FACTOR * best_q * (1 - dones)
        q_values[np.arange(BATCH_SIZE), actions] = targets
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def save(self, episode):
        os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)

        if BACKEND == 'torch':
            path = f'models_{VERSION_NAME}/dqn_{episode}.pth'
            torch.save(self.model.state_dict(), path)
        else:
            path = f'models_{VERSION_NAME}/dqn_{episode}.keras'
            self.model.save(path)

        print(f'Model saved → {path}')

    def save_memory(self, episode):
        os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)
        file = f'models_{VERSION_NAME}/memory_{episode}'
        mem = list(self.memory.buffer)
        np.save(file, np.array(mem, dtype=object))
        print(f'{file}.npy saved')

    def load(self, episode):
        if BACKEND == 'torch':
            path = f'models_{VERSION_NAME}/dqn_{episode}.pth'
            self.model.load_state_dict(torch.load(path, map_location='cpu'))
            self.model.eval()
        else:
            path = f'models_{VERSION_NAME}/dqn_{episode}.keras'
            from tensorflow import keras as _k
            self.model = _k.models.load_model(path)

        print(f'Model loaded ← {path}')


def get_explore_rate(episode):
    return max(
        MIN_EXPLORE_RATE,
        min(
            MAX_EXPLORE_RATE,
            1.0 - math.log10((episode + 1) / DECAY_FACTOR)
        )
    )


def simulate(agent, learning=True, episode_start=0):
    explore_rate = get_explore_rate(episode_start)
    total_rewards = []
    max_reward = -10_000
    total_reward = 0
    env.set_view(True)

    for episode in range(episode_start, NUM_EPISODES + episode_start):

        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.xlabel('episodes')
                plt.title('DQN v2 Training Rewards')
                plt.show(block=False)
                plt.pause(4.0)
                agent.save(episode)
                agent.save_memory(episode)
                np.save(f'models_{VERSION_NAME}/rewards_{episode}.npy', np.array(total_rewards))
                plt.close()

        obv, _ = env.reset()
        state = np.array(obv, dtype=np.float32)
        total_reward = 0

        if not learning:
            env.pyrace.mode = 2

        for t in range(MAX_T):
            action = agent.select_action(state, explore_rate if learning else 0.0)

            obv, reward, done, _, info = env.step(action)
            next_state = np.array(obv, dtype=np.float32)

            if learning:
                agent.memory.push(state, action, reward, next_state, float(done))
                agent.learn()

            total_reward += reward
            state = next_state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs([
                    'DQN V2 SIMULATE',
                    f'Episode: {episode}',
                    f'Time steps: {t}',
                    f'check: {info["check"]}',
                    f'dist: {info["dist"]:.2f}',
                    f'crash: {info["crash"]}',
                    f'Reward: {total_reward:.2f}',
                    f'Max Reward: {max_reward:.2f}',
                    f'Explore: {explore_rate:.3f}',
                    f'Buffer: {len(agent.memory)}'
                ])
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        explore_rate = get_explore_rate(episode)


def load_and_play(agent, episode, learning=False):
    agent.load(episode)
    simulate(agent, learning=learning, episode_start=episode)


if __name__ == '__main__':
    # Part 2 uses the improved environment:
    # - improved reward shaping
    # - brake action
    # - continuous radar observations
    env = gym.make('Pyrace-v2').unwrapped
    print('env', type(env))
    os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)

    STATE_SIZE = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n
    print(f'State size: {STATE_SIZE}, Action size: {ACTION_SIZE}')

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    # simulate(agent, learning=True)
    # load_and_play(agent, 2500, learning=True)
    load_and_play(agent, 7000, learning=False)