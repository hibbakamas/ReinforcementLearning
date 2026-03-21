import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

import gymnasium as gym
import gym_race

# ── Deep Learning backend: try PyTorch, fall back to Keras/TF ──────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    BACKEND = 'torch'
except ImportError:
    from tensorflow import keras
    from tensorflow.keras import layers
    BACKEND = 'keras'

# ─────────────────────────────────────────────────────────────────────────────
VERSION_NAME = 'DQN_v01'

REPORT_EPISODES  = 500
DISPLAY_EPISODES = 100

# ── Hyper-parameters ──────────────────────────────────────────────────────────
DISCOUNT_FACTOR  = 0.99
LEARNING_RATE    = 1e-3
BATCH_SIZE       = 64
MEMORY_SIZE      = 50_000   # replay-buffer capacity
MIN_MEMORY       = 1_000    # start learning only after this many transitions
MIN_EXPLORE_RATE = 0.01
MAX_EXPLORE_RATE = 1.0
DECAY_FACTOR     = 500      # episodes for epsilon decay  (log-schedule)
NUM_EPISODES     = 65_000
MAX_T            = 2_000


# ═══════════════════════════════════════════════════════════════════════════════
#  Neural-network definitions (one per backend)
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_torch(state_size, action_size):
    """Simple MLP: state → 64 → 64 → actions  (PyTorch)"""
    model = nn.Sequential(
        nn.Linear(state_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, action_size),
    )
    return model


def build_model_keras(state_size, action_size):
    """Simple MLP: state → 64 → 64 → actions  (Keras)"""
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(action_size, activation='linear'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse')
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Replay buffer  (replaces the flat np-array "memory" from the Q-Table code)
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=MEMORY_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
#  DQN Agent  (wraps model + replay buffer + training step)
# ═══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    "Vanilla" DQN  –  single network, experience-replay, ε-greedy exploration.
    No target network is required (but adding one is trivial: copy self.model
    weights every N steps into a self.target_model and use that for best_q).
    """

    def __init__(self, state_size, action_size):
        self.state_size  = state_size
        self.action_size = action_size
        self.memory      = ReplayBuffer(MEMORY_SIZE)

        if BACKEND == 'torch':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model  = build_model_torch(state_size, action_size).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
            self.loss_fn   = nn.MSELoss()
        else:
            self.model = build_model_keras(state_size, action_size)

    # ── action selection ──────────────────────────────────────────────────────
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

    # ── single gradient update on a mini-batch ────────────────────────────────
    def learn(self):
        if len(self.memory) < MIN_MEMORY:
            return  # wait until the buffer has enough transitions

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

        # Current Q-values for the taken actions
        q_pred = self.model(S).gather(1, A.unsqueeze(1)).squeeze(1)

        # Target: r  +  γ · max_a' Q(s', a')   (0 if terminal)
        with torch.no_grad():
            best_q = self.model(S2).max(1)[0]
        q_target = R + DISCOUNT_FACTOR * best_q * (1 - D)

        loss = self.loss_fn(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _learn_keras(self, states, actions, rewards, next_states, dones):
        q_values      = self.model.predict(states, verbose=0)
        q_next        = self.model.predict(next_states, verbose=0)
        best_q        = np.max(q_next, axis=1)
        targets       = rewards + DISCOUNT_FACTOR * best_q * (1 - dones)
        q_values[np.arange(BATCH_SIZE), actions] = targets
        self.model.fit(states, q_values, epochs=1, verbose=0)

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, episode):
        os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)
        if BACKEND == 'torch':
            path = f'models_{VERSION_NAME}/dqn_{episode}.pth'
            torch.save(self.model.state_dict(), path)
        else:
            path = f'models_{VERSION_NAME}/dqn_{episode}.keras'
            self.model.save(path)
        print(f'Model saved → {path}')

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Exploration schedule  (mirrors the Q-Table version)
# ═══════════════════════════════════════════════════════════════════════════════

def get_explore_rate(episode):
    return max(MIN_EXPLORE_RATE,
               min(MAX_EXPLORE_RATE,
                   1.0 - math.log10((episode + 1) / DECAY_FACTOR)))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main training / simulation loop
# ═══════════════════════════════════════════════════════════════════════════════

def simulate(agent, learning=True, episode_start=0):
    """
    Main loop – structure mirrors the original Q-Table simulate().
    Key changes:
      • state  : raw float observation vector  (no bucketing needed)
      • update : agent.learn() on a mini-batch  (instead of Q-table update)
      • memory : agent.memory.push()            (instead of env.remember())
    """
    explore_rate  = get_explore_rate(episode_start)
    total_rewards = []
    max_reward    = -10_000
    total_reward  = 0
    env.set_view(True)

    for episode in range(episode_start, NUM_EPISODES + episode_start):

        if episode > 0:
            total_rewards.append(total_reward)          # noqa – set in loop below

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(4.0)
                agent.save(episode)
                plt.close()

        obv, _       = env.reset()
        state        = np.array(obv, dtype=np.float32)  # ← raw floats, no bucket
        total_reward = 0

        if not learning:
            env.pyrace.mode = 2  # continuous display

        for t in range(MAX_T):
            action = agent.select_action(state, explore_rate if learning else 0.0)

            obv, reward, done, _, info = env.step(action)
            next_state = np.array(obv, dtype=np.float32)

            # Store transition and learn
            agent.memory.push(state, action, reward, next_state, float(done))
            if learning:
                agent.learn()

            total_reward += reward
            state         = next_state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(['DQN SIMULATE',
                              f'Episode: {episode}',
                              f'Time steps: {t}',
                              f'check: {info["check"]}',
                              f'dist: {info["dist"]}',
                              f'crash: {info["crash"]}',
                              f'Reward: {total_reward:.0f}',
                              f'Max Reward: {max_reward:.0f}',
                              f'Explore: {explore_rate:.3f}',
                              f'Buffer: {len(agent.memory)}'])
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        explore_rate = get_explore_rate(episode)


def load_and_play(agent, episode, learning=False):
    agent.load(episode)
    simulate(agent, learning=learning, episode_start=episode)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    env = gym.make('Pyrace-v1').unwrapped
    print('env', type(env))
    os.makedirs(f'models_{VERSION_NAME}', exist_ok=True)

    STATE_SIZE  = env.observation_space.shape[0]   # 5 continuous radar readings
    ACTION_SIZE = env.action_space.n               # 3 discrete actions
    print(f'State size: {STATE_SIZE}, Action size: {ACTION_SIZE}')

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)

    # ── Choose one of the lines below ────────────────────────────────────────
    # simulate(agent, learning=True)           # Train from scratch
    load_and_play(agent, 10000, learning=False)  # Play with a saved model
    # load_and_play(agent, 4000, learning=True)   # Continue training