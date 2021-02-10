import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from environments import init_path
import os
import num2words
from .centipede_env import CentipedeEnv


class CpCentipedeFourEnv(CentipedeEnv):
    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4, is_crippled=True)


class CpCentipedeSixEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6, is_crippled=True)


class CpCentipedeEightEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8, is_crippled=True)


class CpCentipedeTenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10, is_crippled=True)


class CpCentipedeTwelveEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12, is_crippled=True)


class CpCentipedeFourteenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14, is_crippled=True)


# regular


class CentipedeFourEnv(CentipedeEnv):
    def __init__(self):
        super(CentipedeFourEnv, self).__init__(CentipedeLegNum=4)


class CentipedeSixEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=6)


class CentipedeEightEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=8)


class CentipedeTenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=10)


class CentipedeTwelveEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=12)


class CentipedeFourteenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=14)


class CentipedeEighteenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=18)


class CentipedeTwentyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=20)


class CentipedeTwentyFourEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=24)


class CentipedeThirtyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=30)


class CentipedeFortyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=40)


class CentipedeFiftyEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=50)


class CentipedeOnehundredEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=100)


"""
    the following environments are models with odd number legs
"""


class CentipedeThreeEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=3)


class CentipedeFiveEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=5)


class CentipedeSevenEnv(CentipedeEnv):
    def __init__(self):
        CentipedeEnv.__init__(self, CentipedeLegNum=7)
