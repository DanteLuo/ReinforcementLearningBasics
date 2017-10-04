from gym.envs.registration import register


register(
    id="FrozenLakeLarge-v0",
    entry_point="frozen_lakes.lake_envs:FrozenLakeLargeEnv",
)

register(
    id="FrozenLakeLargeShiftedIce-v0",
    entry_point="frozen_lakes.lake_envs:FrozenLakeLargeShiftedIceEnv",
)
