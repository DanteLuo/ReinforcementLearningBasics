from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


class FrozenLakeLargeEnv(FrozenLakeEnv):

    def __init__(self):
        desc = ["SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHHFFF",
                "FHHFFFHH",
                "FHFFHFHF",
                "FFFHFFFG"]
        super(FrozenLakeLargeEnv, self).__init__(desc=desc)


    def step(self, action):
        observation, reward, done, info = \
            super(FrozenLakeLargeEnv, self)._step(action)
        if reward == 0 and done:
            reward = -1
        return observation, reward, done, info



class FrozenLakeLargeShiftedIceEnv(FrozenLakeEnv):

    def __init__(self):
        desc = ["SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG"]
        super(FrozenLakeLargeEnv, self).__init__(desc=desc)


    def _step(self, action):
        observation, reward, done, info = \
            super(FrozenLakeLargeShiftedIceEnv, self).step(action)
        if reward == 0 and done:
            reward = -1
        return observation, reward, done, info
