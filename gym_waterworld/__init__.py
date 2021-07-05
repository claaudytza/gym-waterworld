from gym.envs.registration import register

register(
    id='waterworld-v0',
    entry_point='gym_waterworld.envs:WaterworldEnv',
)

# register(
#     id='waterworld-extrahard-v0',
#     entry_point='gym_waterworld.envs:WaterworldExtraHardEnv',
# )