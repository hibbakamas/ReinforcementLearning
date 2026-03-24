# RLI Assignment 17 – Pit Lane Repairs
### IE University – Reinforcement Learning Introduction
**Due: April 7, 2026**

---

## Overview

This project converts a Q-Table racing agent into a Deep Q-Network (DQN) agent, then improves it through reward engineering and environment modifications. It is split into three parts:

| Part | Points | Status |
|------|--------|--------|
| Part 1 – Vanilla DQN | 60% | ✅ Done |
| Part 2 – Improvements + Analysis | 40% | ✅ Done |
| Bonus – Stable Baselines 3 | +30% | Optional |

---

## Project Structure

```
RLI_17_A0/
│
├── Pyrace_RL_DQN.py                      # Part 1 – Vanilla DQN agent
├── Pyrace_RL_DQN_v2.py                   # Part 2 – Improved DQN agent
├── Pyrace_DQN_performance_analysis.ipynb # Part 2 – Analysis notebook
│
├── gym_race/
│   └── envs/
│       ├── __init__.py                   # Registers Pyrace-v1 and Pyrace-v2
│       ├── race_env.py                   # Gym wrappers (RaceEnv + RaceEnvV2)
│       ├── pyrace_2d.py                  # Game logic (MODIFIED for Part 2)
│       └── utils.py                      # Helper functions
│
├── models_DQN_v01/                       # Saved checkpoints from Part 1 training
│   ├── dqn_500.pth                       # Model weights at episode 500
│   ├── dqn_1000.pth                      # Model weights at episode 1000
│   ├── memory_500.npy                    # Replay buffer at episode 500
│   └── ...                              # etc.
│
├── models_DQN_v02/                       # Saved checkpoints from Part 2 training
│   └── ...
│
├── race_track_ie.png                     # Track image
├── car.png                               # Car sprite
└── README.md                             # This file
```

---

## Part 1 – Vanilla DQN (60 pts)

**File:** `Pyrace_RL_DQN.py`

Converts the original Q-Table agent (`Pyrace_RL_QTable.py`) into a neural network-based DQN. The network replaces the lookup table with a small MLP (5 inputs → 64 → 64 → 3 outputs).

**Key changes from Q-Table:**
- Q-values are approximated by a neural network instead of a table
- Uses experience replay (replay buffer of 50,000 transitions)
- Trains on random mini-batches of 64 transitions per step
- State is still the 5 bucketed radar readings from the original env (`Pyrace-v1`)

**What it generates:**
- `models_DQN_v01/dqn_<episode>.pth` — model weights saved every 500 episodes
- `models_DQN_v01/memory_<episode>.npy` — replay buffer saved every 500 episodes

### How to run Part 1

```bash
python Pyrace_RL_DQN.py
```

At the bottom of the file, choose one mode:

```python
simulate(agent, learning=True)              # Train from scratch
load_and_play(agent, 10000, learning=False) # Watch a saved model play
load_and_play(agent, 10000, learning=True)  # Continue training from checkpoint
```

---

## Part 2 – Improvements + Analysis (40 pts)

### 2a – Improved Environment

**Modified file:** `gym_race/envs/pyrace_2d.py`

Three improvements were made to the environment:

| Change | Original | Improved |
|--------|----------|----------|
| Reward | Sparse (0 every step) | Dense (+1/-1 per step toward checkpoint, +500 per checkpoint) |
| State inputs | Bucketed integers (÷20) | Continuous floats (raw px ÷ 20.0) |
| Actions | 3 (accelerate, left, right) | 4 (+ BRAKE) |

The improved environment is registered as `Pyrace-v2` in `__init__.py`.

**Files changed:**
- `gym_race/envs/pyrace_2d.py` — reward function, observe(), action()
- `gym_race/envs/race_env.py` — added `RaceEnvV2` class with float observation space and 4 actions
- `gym_race/envs/__init__.py` — added `Pyrace-v2` registration

### 2b – Improved DQN Agent

**File:** `Pyrace_RL_DQN_v2.py`

Same DQN architecture as Part 1, but trained on `Pyrace-v2` instead of `Pyrace-v1`. Saves to `models_DQN_v02/`.

### How to run Part 2

```bash
python Pyrace_RL_DQN_v2.py
```

### 2c – Analysis Notebook

**File:** `Pyrace_DQN_performance_analysis.ipynb`

Compares the original (v1) and improved (v2) agents side by side:
1. Learning curves for both agents
2. Direct normalised comparison plot
3. Policy heatmaps (greedy action over radar space)
4. Q-value profiles per radar beam
5. Summary statistics table
6. Written discussion of why the improvements worked

**To run the notebook:**
1. Make sure both training runs have completed and checkpoints exist
2. Update the episode numbers at the top of the notebook:
```python
EPISODE_ORIG = 10000   # your v1 checkpoint
EPISODE_NEW  = 10000   # your v2 checkpoint
```
3. Run all cells top to bottom
4. Fill in the `[DESCRIBE]` and `[FILL IN]` placeholders with observations from the actual output charts

---

## Bonus – Stable Baselines 3 (optional, +30 pts)

Migrates the improved environment to a more advanced RL algorithm (PPO or DDPG) using the Stable Baselines 3 library.

```bash
pip install stable-baselines3
```

Since `Pyrace-v2` is already Gymnasium-compatible, SB3 can be used directly:

```python
from stable_baselines3 import PPO
import gymnasium as gym
import gym_race

env = gym.make("Pyrace-v2")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## Setup & Installation

### Requirements

```bash
pip install gymnasium pygame torch numpy matplotlib scipy pandas
```

Or with Keras/TensorFlow instead of PyTorch:

```bash
pip install gymnasium pygame tensorflow numpy matplotlib scipy pandas
```

### First run

Make sure these files are in the project root:
- `race_track_ie.png`
- `car.png`

Then run either training script:

```bash
python Pyrace_RL_DQN.py      # Part 1
python Pyrace_RL_DQN_v2.py   # Part 2
```

---

## Environment Summary

| Env ID | File | Reward | States | Actions |
|--------|------|--------|--------|---------|
| `Pyrace-v1` | `RaceEnv` | Sparse | Integers 0–10 | 3 |
| `Pyrace-v2` | `RaceEnvV2` | Dense | Floats 0.0–10.0 | 4 |

**Actions:**
- `0` — Accelerate (speed +2)
- `1` — Turn Left (angle +5°)
- `2` — Turn Right (angle -5°)
- `3` — Brake (speed -2) ← new in v2

**State (5 radar beams):**
Radar distances measured at -90°, -45°, 0°, +45°, +90° relative to the car's heading.

---

## Notes

- Model files (`.pth`, `.npy`) are large and excluded from git via `.gitignore`
- The `total_reward = 0` initialisation at the top of `simulate()` is important — without it `load_and_play()` crashes on the first episode
- Training speed depends heavily on hardware — expect ~1,000 episodes per few minutes on CPU