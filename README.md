# Pyrace Assignment – Reinforcement Learning

This repository contains our full implementation for the Pyrace assignment, including:

- **Part 1:** DQN agent on the original environment
- **Part 2:** Comparison between V1 and V2 environments
- **Bonus:** PPO experiment using Stable-Baselines3

---

## Project Overview

The goal of this assignment is to train an agent to drive a car in a custom 2D racing environment using reinforcement learning.

We implemented:
- A **Deep Q-Network (DQN)** agent
- An improved version of the environment (V2)
- A comparison between both versions
- A bonus PPO experiment

---

## Main files to check

### 1. Main notebook (MOST IMPORTANT)

**`Pyrace_V1_V2_Comparison_and_Bonus.ipynb`**

This contains:
- explanation of Part 1 (baseline DQN)
- explanation of Part 2 modifications
- training comparison (plots)
- evaluation results
- PPO bonus experiment
- final conclusions

---

## Part 1 (Baseline DQN)

### Files:

**`Pyrace_RL_DQN.py`**

Contains:
- DQN agent implementation
- replay buffer
- epsilon-greedy exploration
- training loop (`simulate`)
- model saving/loading

### Environment used:
- `Pyrace-v1`

### Key characteristics:
- discrete radar observations
- 3 actions:
  - accelerate
  - turn left
  - turn right
- sparse reward:
  - large reward for goal
  - large penalty for crash

---

## Part 2 (V1 vs V2 Comparison)

### Files:

**`Pyrace_RL_DQN_v2.py`**  
Same DQN structure, but trained on the modified environment.

---

### Environment changes (V2)

Implemented in:

**`gym_race/envs/pyrace_2d.py`**  
**`gym_race/envs/race_env.py`**

#### Changes:

**Observations**
- V1: discretized radar values (integers)
- V2: continuous radar values (floats)

**Actions**
- V1: 3 actions
- V2: 4 actions (added brake)

**Reward function (major change)**
- checkpoint bonus
- progress-based reward
- small step penalty
- crash penalty
- goal reward

This is the main reason V2 performs better.

---

## Environment structure

### Core files:

**`gym_race/envs/pyrace_2d.py`**
- car physics
- radar sensors
- reward logic
- V1 and V2 implementations

**`gym_race/envs/race_env.py`**
- Gym wrapper
- defines:
  - `RaceEnv` → V1
  - `RaceEnvV2` → V2

**`gym_race/__init__.py` + `envs/__init__.py`**
- environment registration
- allows:
  - `gym.make('Pyrace-v1')`
  - `gym.make('Pyrace-v2')`

---

## Saved training artifacts

We include the 5000-episode checkpoints used in the notebook:

- `models_DQN_v01/dqn_5000.pth`
- `models_DQN_v01/rewards_5000.npy`
- `models_DQN_v02/dqn_5000.pth`
- `models_DQN_v02/rewards_5000.npy`

These are used for:
- plotting training curves
- evaluation without retraining

---

## Bonus (PPO)

We also tested a **PPO agent** using Stable-Baselines3.

### What was done:
- trained PPO on `Pyrace-v2`
- evaluated using the same evaluation setup
- compared results against DQN

### Result:
- PPO improved over V1
- but did **not outperform V2 DQN**

---

## How to run

### Train V1:
```bash
python Pyrace_RL_DQN.py
```

### Train V2:
python Pyrace_RL_DQN_v2.py

### Run notebook:

Open:

Pyrace_V1_V2_Comparison_and_Bonus.ipynb

### Suggested reading order
1. Notebook (main explanation)
2. Pyrace_RL_DQN.py (Part 1)
3. Pyrace_RL_DQN_v2.py (Part 2)
4. pyrace_2d.py (environment logic)
5. race_env.py (Gym wrapper)


#### Final conclusions
- V1 struggled due to sparse rewards and limited actions
- V2 significantly improved performance due to:
    - better reward shaping
    - richer observations
    - additional control (brake)
- PPO performed reasonably well but did not exceed V2 DQN

Hence, the biggest improvement came from environment design, not just the algorithm