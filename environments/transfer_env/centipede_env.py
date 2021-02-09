#!/usr/bin/env python2
# -----------------------------------------------------------------------------
#   @brief:
#       The Centipede environments.
#   @author:
#       Tingwu (Wilson) Wang, Aug. 30nd, 2017
# -----------------------------------------------------------------------------

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from util import init_path
import os
import num2words


class CentipedeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        @brief:
            In the CentipedeEnv, we define a task that is designed to test the
            power of transfer learning for gated graph neural network.
        @children:
            @CentipedeFourEnv
            @CentipedeEightEnv
            @CentipedeTenEnv
            @CentipedeTwelveEnv
    """

    def __init__(self, CentipedeLegNum=4, is_crippled=False):

        # get the path of the environments
        if is_crippled:
            xml_name = "CpCentipede" + self.get_env_num_str(CentipedeLegNum) + ("" if tt else "TT") + ".xml"
        else:
            xml_name = "Centipede" + self.get_env_num_str(CentipedeLegNum) + ".xml"
        xml_path = os.path.join(
            init_path.get_base_dir(), "environments", "assets", xml_name
        )
        xml_path = str(os.path.abspath(xml_path))
        self.num_body = int(np.ceil(CentipedeLegNum / 2.0))
        self._control_cost_coeff = 0.5 * 4 / CentipedeLegNum
        self._contact_cost_coeff = 0.5 * 1e-3 * 4 / CentipedeLegNum

        self._n_legs = CentipedeLegNum

        self.torso_geom_id = 1 + np.array(list(range(self.num_body))) * 5
        # make sure the centipede is not born to be end of episode
        self.body_qpos_id = 6 + 6 + np.array(list(range(self.num_body))) * 6
        self.body_qpos_id[-1] = 5

        mujoco_env.MujocoEnv.__init__(self, xml_path, 5)
        utils.EzPickle.__init__(self)

    def get_env_num_str(self, number):
        num_str = num2words.num2words(number).replace("-", "")
        return num_str[0].upper() + num_str[1:]

    def step(self, a):
        xposbefore = self.get_body_com("torso_" + str(self.num_body - 1))[0]
        self.do_simulation(a, self.frame_skip)
        """
        xposafter = np.mean([self.get_body_com("torso_" + str(i_torso))[0]
                             for i_torso in range(self.num_body)])
        """
        xposafter = self.get_body_com("torso_" + str(self.num_body - 1))[0]

        # calculate reward
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = self._control_cost_coeff * np.square(a).sum()

        contact_cost = self._contact_cost_coeff * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        # check if finished
        state = self.state_vector()
        notdone = (
            np.isfinite(state).all()
            and self._check_height()
            and self._check_direction()
        )
        done = not notdone

        ob = self._get_obs()

        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        VEL_NOISE_MULT = 0.1
        POS_NOISE_MULT = 0.1
        while True:
            qpos = self.init_qpos + self.np_random.uniform(
                size=self.model.nq, low=-POS_NOISE_MULT, high=POS_NOISE_MULT
            )
            qpos[self.body_qpos_id] = self.np_random.uniform(
                size=len(self.body_qpos_id),
                low=-POS_NOISE_MULT / (self.num_body - 1),
                high=POS_NOISE_MULT / (self.num_body - 1),
            )

            qvel = self.init_qvel + self.np_random.randn(self.model.nv) * VEL_NOISE_MULT
            self.set_state(qpos, qvel)
            if self._check_height() and self._check_direction():
                break

        return self._get_obs()

    def viewer_setup(self):
        # self.viewer.mode = 'rgb_array'
        # self.viewer.cam.distance = self.model.stat.extent * 3
        # body_name = 'torso_' + str(int(np.ceil(self.num_body / 2 - 1)))
        # #self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)
        # self.viewer.cam.trackbodyid = 4
        # #self.viewer.cam.lookat[2] = self.model.stat.center[2]
        # self.viewer.cam.elevation = -20
        # self.viewer.cam.azimuth = -20

        # self.viewer.cam.distance = self.model.stat.extent * 5.
        body_name = "torso_" + str(int(np.ceil(self.num_body / 2 - 1)))
        # self.viewer.cam.trackbodyid = self.model.body_names.index(body_name)
        # self.viewer.cam.trackbodyid = 10

    """
    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.5).all() and (height > 0.35).all()
    """

    def _check_height(self):
        height = self.data.geom_xpos[self.torso_geom_id, 2]
        return (height < 1.15).all() and (height > 0.35).all()

    def _check_direction(self):
        y_pos_pre = self.data.geom_xpos[self.torso_geom_id[:-1], 1]
        y_pos_post = self.data.geom_xpos[self.torso_geom_id[1:], 1]
        y_diff = np.abs(y_pos_pre - y_pos_post)
        return (y_diff < 0.45).all()


"""
    the following environments are just models with even number legs
"""


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


class CentipedeTTSixEnv(CentipedeEnv):
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


class CentipedeTwentyfourEnv(CentipedeEnv):
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


DEFAULT_MULTITASK_CENTIPEDES = [6, 8]


class CentipedeContainer(CentipedeEnv):
    def __init__(self, leg_types=None):
        if leg_types is None:
            leg_types = DEFAULT_MULTITASK_CENTIPEDES
        self._envs = [CentipedeEnv(n_legs) for n_legs in leg_types]

        self._curr_env = np.random.choice(self._envs)

    def reset_model(self):
        self._curr_env = np.random.choice(self._envs)
        return self._curr_env.reset()

    def step(self, a):
        return self._curr_env.step(a)

    @property
    def observation_space(self):
        return self._curr_env.observation_space

    @property
    def action_space(self):
        return self._curr_env.action_space

    @property
    def sim(self):
        return self._curr_env.sim

    @property
    def ob_normalizer(self):
        return self._curr_env.ob_normalizer
