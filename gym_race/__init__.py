from gymnasium.envs.registration import register

# Original environment – Q-Table / vanilla DQN (Part 1)
register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)

# Improved environment – dense reward, continuous states, brake action (Part 2)
register(
    id='Pyrace-v2',
    entry_point='gym_race.envs:RaceEnvV2',
    max_episode_steps=2_000,
)